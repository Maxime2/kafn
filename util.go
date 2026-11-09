package kafn

import "math"

// ArgMax is the index of the largest element
func ArgMax(xx []float64) int {
	if len(xx) == 0 {
		return -1
	}
	max, idx := xx[0], 0
	for i, x := range xx {
		if x > max {
			max, idx = x, i
		}
	}
	return idx
}

// Sum is sum
func Sum(xx []float64) (sum float64) {
	for _, x := range xx {
		sum += x
	}
	return
}

// Round to nearest integer
func Round(x float64) float64 {
	return DF(math.Floor(float64(x) + .5))
}

// fibonacci returns a function that returns
// successive fibonacci numbers from each
// successive call
func Fibonacci() func() int {
	first, second := 1, 2
	return func() int {
		ret := first
		first, second = second, first+second
		return ret
	}
}
