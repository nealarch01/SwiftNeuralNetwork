//
//  Neuron.swift
//  
//
//  Created by Neal Archival on 5/20/23.
//

import Foundation

class Neuron: Codable {
    var collector: Double
    var delta: Double
    var weights: [Double]

    init(collector: Double = 0.0) {
        self.collector = collector
        self.delta = 0.0
        self.weights = []
    }

    func addConnection(neuron: Neuron, weight: Double = Double.random(in: 0.0...1.0)) {
        self.weights.append(weight)
    }

    func updateCollector(newCollector: Double) {
        self.collector = newCollector
    }

    func updateDelta(newDelta: Double) {
        self.delta = newDelta
    }
    
    func updateWeights(index: Int, newWeight: Double) {
        self.weights[index] = newWeight
    }
}
