import Foundation

public struct NeuralNetwork {
    private(set) var layers: [[Neuron]] = []
    
    public init?(layerStructure: [Int]) {
        if layerStructure.count < 2 {
            print("Error: Network must have at least 2 layers")
            return nil
        }
        for neuronCount in layerStructure {
            if neuronCount < 1 {
                print("Error: ")
                return nil
            }
            let newLayer = createLayer(size: neuronCount)
            self.layers.append(newLayer)
        }
    }
    
    mutating public func train(trainingInputs: [[Double]], expectedOutputs: [[Double]], learningRate: Double, targetError: Double, epochs: Int) {
        var trainingInputsCpy = trainingInputs
        var expectedOutputsCpy = expectedOutputs
        for epoch in 0..<epochs {
            var sumError = 0.0
            for i in 0..<trainingInputsCpy.count {
                setInputLayer(row: &trainingInputsCpy[i])
                propagateForward()
                let outputs = self.layers[self.layers.count - 1].map { $0.collector }
                for j in 0..<outputs.count {
                    let error = expectedOutputs[i][j] - outputs[j]
                    sumError += pow(error, 2)
                }
                self.propagateBackward(expected: &expectedOutputsCpy[i]) // TODO: Fix this
                self.adjustWeights(learningRate: learningRate)
            }
            if sumError <= targetError {
                print("Target error reached!")
                print("> epoch: \(epoch), learning rate: \(learningRate), error: \(sumError)")
                return
            }
            print("> epoch: \(epoch), learning rate: \(learningRate), error: \(sumError)")
        }
    }
    
    public func testLayerSummation(inputs: [Double]) {
        let layerSum: (_ index: Int) -> Double = { index in
            var sum = 0.0
            for i in 0..<layers[index].count {
                sum += layers[index][i].collector
            }
            return sum
        }
        var inputsCpy = inputs
        self.setInputLayer(row: &inputsCpy)
        var total = layerSum(0)
        // Feed forward with the inputs
        for i in 1..<layers.count {
            for j in 0..<layers[i].count {
                layers[i][j].updateCollector(newCollector: total)
            }
            total = layerSum(i)
        }
    }
    
    public func getLayerCollectors(index: Int) -> [Double] {
        if index >= layers.count {
            print("Error: Invalid Index")
            return []
        }
        return layers[index].map { $0.collector }
    }
    
    private func propagateForward() {
        // Start after the input layer
        for layerIndex in 1..<self.layers.count { // Iterate through each layer
            for neuronIndex in 0..<self.layers[layerIndex].count { // Get the current neuron for an index to the edge weight
                var weightedSum = 0.0
                for previousNeuronIndex in 0..<self.layers[layerIndex - 1].count { // Look back to previous layer node
                    let collectorValue = self.layers[layerIndex - 1][previousNeuronIndex].collector
                    let edgeWeight = self.layers[layerIndex - 1][previousNeuronIndex].weights[neuronIndex]
                    weightedSum += (edgeWeight * collectorValue)
                }
                self.layers[layerIndex][neuronIndex].updateCollector(newCollector: transferActivation(weightedSum))
            }
        }
    }
    
    private func propagateBackward(expected: inout [Double]) {
        for layerIndex in (0..<self.layers.count).reversed() {
            var layerErrors: [Double] = []
            if layerIndex != layers.count - 1 { // True condition runs the fastest
                // We are not at the output layer
                for nodeIndex in 0..<self.layers[layerIndex].count {
                    var error: Double = 0.0
                    for weightIndex in 0..<self.layers[layerIndex + 1].count {
                        let weight = self.layers[layerIndex][nodeIndex].weights[weightIndex]
                        let delta = self.layers[layerIndex + 1][weightIndex].delta
                        error += (weight * delta)
                    }
                    layerErrors.append(error)
                }
            } else {
                // We are at the output layer
                for nodeIndex in 0..<self.layers[layerIndex].count {
                    let collector = self.layers[layerIndex][nodeIndex].collector
                    let error = collector - expected[nodeIndex]
                    layerErrors.append(error)
                }
            } // else
            // Update the delta
            for nodeIndex in 0..<self.layers[layerIndex].count {
                let oldDelta = self.layers[layerIndex][nodeIndex].delta
                let delta = layerErrors[nodeIndex] * transferDerivative(self.layers[layerIndex][nodeIndex].collector)
                self.layers[layerIndex][nodeIndex].updateDelta(newDelta: delta)
            }
        }
    }
    
    private func adjustWeights(learningRate: Double) {
        // General formula: weight = weight - learningRate * delta * collectorFromPreviousLayer
        for layerIndex in 1..<self.layers.count {
            let inputCollectors: [Double] = self.getLayerCollectors(index: layerIndex - 1)
            for currentLayerNodeIndex in 0..<self.layers[layerIndex].count {
                for prevLayerNodeIndex in 0..<self.layers[layerIndex - 1].count {
                    let weight = self.layers[layerIndex - 1][prevLayerNodeIndex].weights[currentLayerNodeIndex]
                    let delta = self.layers[layerIndex][currentLayerNodeIndex].delta
                    let collector = inputCollectors[prevLayerNodeIndex]
                    let newWeight = weight - learningRate * delta * collector
                    self.layers[layerIndex - 1][prevLayerNodeIndex].updateWeights(index: currentLayerNodeIndex, newWeight: newWeight)
                }
            }
        }
    }
    
    private func createLayer(size: Int) -> [Neuron] {
        var column: [Neuron] = []
        for _ in 0..<size {
            let neuron = Neuron(collector: 0.0)
            if self.layers.count > 0 {
                for j in 0..<layers[layers.count - 1].count {
                    self.layers[self.layers.count - 1][j].addConnection(neuron: neuron)
                }
            }
            column.append(neuron)
        }
        return column
    }
    
    private func setInputLayer(row: inout [Double]) {
        if row.count < layers[0].count {
            print("Warning: the number of inputs provided does not match the number of input layer neurons. Zero rows have been added")
        }
        for i in 0..<layers[0].count {
            if i >= layers.count {
                layers[0][i].updateCollector(newCollector: 0.0)
                continue
            }
            layers[0][i].updateCollector(newCollector: row[i])
        }
    }
    
    private func transferActivation(_ x: Double) -> Double { // Sigmoidal transfer function
        return 1.0 / (1.0 + exp(-x))
    }
    
    private func transferDerivative(_ x: Double) -> Double {
        return x * (1.0 - x)
    }
    
}
