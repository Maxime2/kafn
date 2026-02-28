package kafn

import (
	"math"
)

type Deepfloat64 = float64

func DF(f float64) Deepfloat64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0.0
	}
	return f
}

func Add(a, b Deepfloat64) Deepfloat64 {
	return a + b
}

func Sub(a, b Deepfloat64) Deepfloat64 {
	return a - b
}

func Mul(a, b Deepfloat64) Deepfloat64 {
	return a * b
}

func Div(a, b Deepfloat64) Deepfloat64 {
	return a / b
}

func Copy(a Deepfloat64) Deepfloat64 {
	return a
}

func Float64(a Deepfloat64) float64 {
	return a
}
