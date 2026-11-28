import Foundation
import Vision

let request = VNDetectAnimalBodyPoseRequest()

func estimate(_ encoded: String) throws -> [[String: [Double]]] {
    guard let image = Data(base64Encoded: encoded) else { return [] }
    try VNImageRequestHandler(data: image, options: [:]).perform([request])
    return try (request.results ?? []).map { observation in
        var joints: [String: [Double]] = [:]
        for (joint, point) in try observation.recognizedPoints(.all) where point.confidence >= 0.25 {
            let name = String(joint.rawValue.rawValue.dropFirst("animal_joint_".count))
            joints[name == "heck" ? "neck" : name] =
                [point.location.x, 1 - point.location.y, Double(point.confidence)]
        }
        return joints
    }
}

while let line = readLine() {
    do {
        let encoded = try JSONDecoder().decode([String].self, from: Data(line.utf8))
        let result = try encoded.map(estimate)
        var data = try JSONEncoder().encode(result)
        data.append(10)
        FileHandle.standardOutput.write(data)
    } catch {
        FileHandle.standardOutput.write(Data("null\n".utf8))
    }
}
