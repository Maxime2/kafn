package kafn

import (
	"math"
)

// GetLoss returns a loss function given a LossType
func GetLoss(loss LossType) Loss {
	switch loss {
	case LossCrossEntropy:
		return CrossEntropy{}
	case LossMeanSquared:
		return MeanSquared{}
	case LossBinaryCrossEntropy:
		return BinaryCrossEntropy{}
	}
	return CrossEntropy{}
}

// LossType represents a loss function
type LossType int

func (l LossType) String() string {
	switch l {
	case LossCrossEntropy:
		return "CE"
	case LossBinaryCrossEntropy:
		return "BinCE"
	case LossMeanSquared:
		return "MSE"
	}
	return "N/A"
}

const (
	// LossNone signifies unspecified loss
	LossNone LossType = 0
	// LossCrossEntropy is cross entropy loss
	LossCrossEntropy LossType = 1
	// LossBinaryCrossEntropy is the special case of binary cross entropy loss
	LossBinaryCrossEntropy LossType = 2
	// LossMeanSquared is MSE
	LossMeanSquared LossType = 3
)

// Loss is satisfied by loss functions
type Loss interface {
	F(estimate, ideal Deepfloat64) Deepfloat64
	Cf(estimate, ideal [][]Deepfloat64) Deepfloat64
	Df(estimate, ideal Deepfloat64) Deepfloat64
}

// CrossEntropy is CE loss
type CrossEntropy struct{}

// F is MSE(...)
func (l CrossEntropy) F(estimate, ideal Deepfloat64) Deepfloat64 {
	d := Sub(estimate, ideal)
	return Div(Mul(d, d), DF(2))
}

// Cf is CE(...)
func (l CrossEntropy) Cf(estimate, ideal [][]Deepfloat64) Deepfloat64 {

	sum := DF(0)
	for i := range estimate {
		ce := DF(0)
		for j := range estimate[i] {
			ce = Add(ce, Mul(ideal[i][j], DF(math.Log(float64(estimate[i][j])))))
		}

		sum = Sub(sum, ce)
	}
	return Div(sum, DF(float64(len(estimate))))
}

// Df is CE'(...)
func (l CrossEntropy) Df(estimate, ideal Deepfloat64) Deepfloat64 {
	return Sub(estimate, ideal)
}

// BinaryCrossEntropy is binary CE loss
type BinaryCrossEntropy struct{}

// F is MSE(...)
func (l BinaryCrossEntropy) F(estimate, ideal Deepfloat64) Deepfloat64 {
	d := Sub(estimate, ideal)
	return Div(Mul(d, d), DF(2))
}

// Cf is CE(...)
func (l BinaryCrossEntropy) Cf(estimate, ideal [][]Deepfloat64) Deepfloat64 {
	epsilon := 1e-16
	sum := DF(0)
	for i := range estimate {
		ce := DF(0)
		for j := range estimate[i] {
			// ce += ideal * log(est+eps) + (1-ideal)*log(1-est+eps)
			term1 := Mul(ideal[i][j], DF(math.Log(float64(estimate[i][j])+epsilon)))
			term2 := Mul(Sub(DF(1), ideal[i][j]), DF(math.Log(1.0-float64(estimate[i][j])+epsilon)))
			ce = Add(ce, Add(term1, term2))
		}
		sum = Sub(sum, ce)
	}
	return Div(sum, DF(float64(len(estimate))))
}

// Df is CE'(...)
func (l BinaryCrossEntropy) Df(estimate, ideal Deepfloat64) Deepfloat64 {
	return Sub(estimate, ideal)
}

// MeanSquared in MSE loss
type MeanSquared struct{}

// F is MSE(...)
func (l MeanSquared) F(estimate, ideal Deepfloat64) Deepfloat64 {
	d := Sub(estimate, ideal)
	return Div(Mul(d, d), DF(2))
}

// Cf is MSE(...)
func (l MeanSquared) Cf(estimate, ideal [][]Deepfloat64) Deepfloat64 {
	sum := DF(0)
	for i := 0; i < len(estimate); i++ {
		for j := 0; j < len(estimate[i]); j++ {
			diff := Sub(estimate[i][j], ideal[i][j])
			sum = Add(sum, Mul(diff, diff))
		}
	}
	return Div(sum, DF(float64(len(estimate)*len(estimate[0]))))
}

// Df is MSE'(...)
func (l MeanSquared) Df(estimate, ideal Deepfloat64) Deepfloat64 {
	return Sub(estimate, ideal)
}
