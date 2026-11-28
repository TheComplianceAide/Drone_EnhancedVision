import Foundation
import Vision
import CoreVideo

private func pixelBuffer(_ data: UnsafePointer<UInt8>, _ w: Int, _ h: Int) -> CVPixelBuffer? {
    var buffer: CVPixelBuffer?
    let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
    let status = CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA,
                                     attributes, &buffer)
    guard status == kCVReturnSuccess, let buffer else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for row in 0..<h {
        memcpy(base.advanced(by: row * stride), data.advanced(by: row * w * 4), w * 4)
    }
    return buffer
}

@_cdecl("generate_vision_flow")
public func generateVisionFlow(
    _ source: UnsafePointer<UInt8>, _ target: UnsafePointer<UInt8>,
    _ width: Int32, _ height: Int32, _ quality: Int32,
    _ vectors: UnsafeMutablePointer<Float>, _ capacity: Int32,
    _ dimensions: UnsafeMutablePointer<Int32>
) -> Int32 {
    autoreleasepool {
        let w = Int(width)
        let h = Int(height)
        guard w >= 32, h >= 32, let from = pixelBuffer(source, w, h), let to = pixelBuffer(target, w, h) else {
            return 1
        }
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: to, options: [:])
        request.revision = VNGenerateOpticalFlowRequestRevision2
        request.computationAccuracy = quality <= 0 ? .low : quality == 1 ? .medium : quality == 2 ? .high : .veryHigh
        request.keepNetworkOutput = true
        do {
            try VNImageRequestHandler(cvPixelBuffer: from, options: [:]).perform([request])
        } catch {
            fputs("Vision optical flow: \(error)\n", stderr)
            return 2
        }
        guard let observation = request.results?.first else { return 3 }
        let result = observation.pixelBuffer
        let fw = CVPixelBufferGetWidth(result)
        let fh = CVPixelBufferGetHeight(result)
        guard fw > 0, fh > 0, fw * fh * 2 <= Int(capacity) else { return 4 }
        dimensions[0] = Int32(fw)
        dimensions[1] = Int32(fh)
        CVPixelBufferLockBaseAddress(result, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(result, .readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(result) else { return 5 }
        let stride = CVPixelBufferGetBytesPerRow(result)
        switch CVPixelBufferGetPixelFormatType(result) {
        case kCVPixelFormatType_TwoComponent32Float:
            for row in 0..<fh {
                memcpy(vectors.advanced(by: row * fw * 2), address.advanced(by: row * stride), fw * 8)
            }
        case kCVPixelFormatType_TwoComponent16Half:
            for row in 0..<fh {
                let pixels = address.advanced(by: row * stride).assumingMemoryBound(to: UInt16.self)
                for col in 0..<(fw * 2) {
                    vectors[row * fw * 2 + col] = Float(Float16(bitPattern: pixels[col]))
                }
            }
        default:
            return 6
        }
        return 0
    }
}
