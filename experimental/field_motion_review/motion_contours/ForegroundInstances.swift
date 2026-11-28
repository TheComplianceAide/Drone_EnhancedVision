import CoreVideo
import Foundation
import Vision

enum MaskError: Error {
    case missingDestination
    case unsupportedPixelFormat(OSType)
    case unreadableBuffer
}

func maskBytes(_ mask: CVPixelBuffer) throws -> Data {
    let format = CVPixelBufferGetPixelFormatType(mask)
    guard format == kCVPixelFormatType_OneComponent8 else {
        throw MaskError.unsupportedPixelFormat(format)
    }
    let width = CVPixelBufferGetWidth(mask)
    let height = CVPixelBufferGetHeight(mask)
    let stride = CVPixelBufferGetBytesPerRow(mask)
    CVPixelBufferLockBaseAddress(mask, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(mask) else {
        throw MaskError.unreadableBuffer
    }
    var output = Data("P5\n\(width) \(height)\n255\n".utf8)
    for y in 0..<height {
        output.append(base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self), count: width)
    }
    return output
}

@available(macOS 14.0, *)
func run() throws {
    var arguments = Array(CommandLine.arguments.dropFirst())
    let streaming = arguments.first == "--stdin"
    if streaming { arguments.removeFirst() }
    guard let folder = arguments.first else { throw MaskError.missingDestination }
    let output = URL(fileURLWithPath: folder, isDirectory: true)
    let request = VNGenerateForegroundInstanceMaskRequest()
    func send(_ message: String) {
        FileHandle.standardOutput.write(Data((message + "\n").utf8))
    }
    func process(_ path: String) {
        let image = URL(fileURLWithPath: path)
        let handler = VNImageRequestHandler(url: image)
        let began = CFAbsoluteTimeGetCurrent()
        do {
            try handler.perform([request])
            let ms = (CFAbsoluteTimeGetCurrent() - began) * 1000
            let name = image.deletingPathExtension().lastPathComponent
            if let observed = request.results?.first, !observed.allInstances.isEmpty {
                let data = try maskBytes(observed.instanceMask)
                try data.write(to: output.appendingPathComponent(name + ".pgm"))
                send("\(name) \(observed.allInstances.count) \(ms)")
            } else {
                send("\(name) 0 \(ms)")
            }
        } catch {
            send("\(image.lastPathComponent) error \(error)")
        }
    }
    if streaming {
        while let path = readLine() { process(path) }
    } else {
        for path in arguments.dropFirst() { process(path) }
    }
}

if #available(macOS 14.0, *) {
    try run()
}
