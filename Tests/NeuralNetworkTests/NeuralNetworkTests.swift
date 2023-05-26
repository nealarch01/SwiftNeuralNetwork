import XCTest
@testable import NeuralNetwork

final class NeuralNetworkTests: XCTestCase {
    func testNeuralNetwork() throws {
        var neuralNetwork = NeuralNetwork(layerStructure: [4, 2, 1])
        XCTAssertNotNil(neuralNetwork, "Neural network failed to initialized given [4, 2, 1]")
        neuralNetwork!.testLayerSummation(inputs: [1, 2, 3])
        
        var layerCollectors = neuralNetwork!.getLayerCollectors(index: 0)
        XCTAssertEqual(layerCollectors[0], 1, "Expected collector == 1")
        XCTAssertEqual(layerCollectors[1], 2, "Expected collector == 2")
        XCTAssertEqual(layerCollectors[2], 3, "Expected collector == 3")
        
        layerCollectors = neuralNetwork!.getLayerCollectors(index: 1)
        XCTAssertEqual(layerCollectors[0], 6, "Expected collector == 6")
        XCTAssertEqual(layerCollectors[1], 6, "Expected collector == 6")
        
        layerCollectors = neuralNetwork!.getLayerCollectors(index: 2)
        XCTAssertEqual(layerCollectors[0], 12, "Expected collector == 12")
        
        print("Layer Test Complete")
        let inputs = [
            [0,0,0,0,0]
        ]
        print("Test run")
        var trainingRows: [[Double]] = []
        var expectedRows: [[Double]] = []
        for input in inputs {
            var inputRow = input.map { Double($0) }
            inputRow.removeLast()
            let expectedRow = [ Double(input[input.count - 1]) ]
            trainingRows.append(inputRow)
            expectedRows.append(expectedRow)
        }
        print("Training Rows:")
        print(trainingRows)
        print("\nExpected rows")
        print(expectedRows)
        print("Starting test train")
        neuralNetwork!.train(
            trainingInputs: trainingRows,
            expectedOutputs: expectedRows,
            learningRate: 0.65,
            targetError: 0.05,
            epochs: 5
        )
        print("Training complete")
        let serializedNetwork = neuralNetwork!.serialized()
        XCTAssertNotNil(serializedNetwork, "Network should be serialized")
        print(serializedNetwork ?? "")
    }
}

