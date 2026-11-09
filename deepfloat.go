package kafn

import (
	"math"
)

func DF(f float64) float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0.0
	}
	return f
}
