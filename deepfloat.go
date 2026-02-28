package kafn

import (
	"math"
	"math/big"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

type Deepfloat64 = *big.Float

func DF(f float64) Deepfloat64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return big.NewFloat(0).SetPrec(tabulatedfunction.Precision)
	}
	return big.NewFloat(f).SetPrec(tabulatedfunction.Precision)
}

func Add(a, b Deepfloat64) Deepfloat64 {
	return new(big.Float).Add(a, b)
}

func Sub(a, b Deepfloat64) Deepfloat64 {
	return new(big.Float).Sub(a, b)
}

func Mul(a, b Deepfloat64) Deepfloat64 {
	return new(big.Float).Mul(a, b)
}

func Div(a, b Deepfloat64) Deepfloat64 {
	return new(big.Float).Quo(a, b)
}

func Copy(a Deepfloat64) Deepfloat64 {
	return new(big.Float).Copy(a)
}

func Float64(a Deepfloat64) float64 {
	f, _ := a.Float64()
	return f
}
