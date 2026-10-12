# src/bloom_filter.py
import mmh3, math

class CountingBloomFilter:
    def __init__(self, capacity=100000, error_rate=0.01):
        self.capacity = capacity
        m = - (capacity * math.log(error_rate)) / (math.log(2) ** 2)
        self.m = int(m)
        k = (self.m / capacity) * math.log(2)
        self.k = int(k) or 1
        self.array = [0] * self.m
    def _hashes(self, key):
        for i in range(self.k):
            yield mmh3.hash(str(key), i) % self.m
    def add(self, key):
        for idx in self._hashes(key):
            self.array[idx] += 1
    def remove(self, key):
        for idx in self._hashes(key):
            if self.array[idx] > 0:
                self.array[idx] -= 1
    def __contains__(self, key):
        return all(self.array[idx] > 0 for idx in self._hashes(key))
