import json
from random import uniform


class Node:
    def __init__(self, name, dependents=None, probabilities=None):
        self.name = name
        self.dependents = dependents
        self.probabilities = probabilities


def load_from_json(file_path):
    with open(file_path, 'r') as file_handle:
        data = json.load(file_handle)

    nodes = []

    for node in data:
        nodes.append(Node(node["name"], node["dependents"], node["probabilities"]))

    return nodes


def generate(nodes, samples_nr):
    total = {'alarm_clock': 0,
             'oversleep': 0,
             'traffic': 0,
             'on_time': 0}
    for _ in range(samples_nr):
        sample = {}
        for node in nodes:
            dependents = ""
            for name in node.dependents:
                dependents += 'T' if sample[name] else 'F'
            sample[node.name] = uniform(0, 1) < node.probabilities[dependents]
            if sample[node.name]:
                total[node.name] += 1

    return total


if __name__ == "__main__":
    nodes = load_from_json('./schema.json')
    repetitions = 10000
    results = generate(nodes, repetitions)

    for key, value in results.items():
        print(f'{key}: {value/repetitions}')
