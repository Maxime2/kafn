package kafn

// Example is an input-target pair
type Example struct {
	Input    []float64
	Response []float64
}

// Examples is a set of input-output pairs
type Examples []Example

// Split assigns each element to two new slices
// according to probability p
func (e Examples) Split(p float64) (first, second Examples) {
	for i := 0; i < len(e); i++ {
		if p > Rand.Float64() {
			first = append(first, e[i])
		} else {
			second = append(second, e[i])
		}
	}
	return
}

// SplitSize splits slice into parts of size size
func (e Examples) SplitSize(size int) []Examples {
	if size <= 0 || len(e) == 0 {
		return nil
	}
	res := make([]Examples, 0)
	for i := 0; i < len(e); i += size {
		res = append(res, e[i:min(i+size, len(e))])
	}
	return res
}

// SplitN splits slice into n parts
func (e Examples) SplitN(n int) []Examples {
	if n <= 0 || len(e) == 0 {
		return nil
	}
	if n > len(e) {
		n = len(e)
	}
	res := make([]Examples, n)
	chunkSize := len(e) / n
	remainder := len(e) % n
	start := 0
	for i := 0; i < n; i++ {
		end := start + chunkSize
		if i < remainder {
			end++
		}
		res[i] = e[start:end]
		start = end
	}
	return res
}
