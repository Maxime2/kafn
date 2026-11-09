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
	F(estimate, ideal float64) float64
	Cf(estimate, ideal [][]float64) float64
}

// CrossEntropy is CE loss
type CrossEntropy struct{}

// F is MSE(...)
func (l CrossEntropy) F(estimate, ideal float64) float64 {
	d := estimate - ideal
	return (d * d) * 0.5
}

// Cf is CE(...)
func (l CrossEntropy) Cf(estimate, ideal [][]float64) float64 {
	var sum float64
	for i := range estimate {
		var ce float64
		for j := range estimate[i] {
			ce += ideal[i][j] * math.Log(estimate[i][j])
		}
		sum -= ce
	}
	return sum / float64(len(estimate))
}

// BinaryCrossEntropy is binary CE loss
type BinaryCrossEntropy struct{}

// F is MSE(...)
func (l BinaryCrossEntropy) F(estimate, ideal float64) float64 {
	d := estimate - ideal
	return (d * d) * 0.5
}

// Cf is CE(...)
func (l BinaryCrossEntropy) Cf(estimate, ideal [][]float64) float64 {
	const epsilon = 1e-16
	var sum float64
	for i := range estimate {
		var ce float64
		for j := range estimate[i] {
			term1 := ideal[i][j] * math.Log(estimate[i][j]+epsilon)
			term2 := (1.0 - ideal[i][j]) * math.Log(1.0-estimate[i][j]+epsilon)
			ce += term1 + term2
		}
		sum -= ce
	}
	return sum / float64(len(estimate))
}

// MeanSquared in MSE loss
type MeanSquared struct{}

// F is MSE(...)
func (l MeanSquared) F(estimate, ideal float64) float64 {
	d := estimate - ideal
	return (d * d) * 0.5
}

// Cf is MSE(...)
func (l MeanSquared) Cf(estimate, ideal [][]float64) float64 {
	var sum float64
	for i := 0; i < len(estimate); i++ {
		estRow := estimate[i]
		idealRow := ideal[i]
		for j := 0; j < len(estRow); j++ {
			diff := estRow[j] - idealRow[j]
			sum += diff * diff
		}
	}
	return sum / float64(len(estimate)*len(estimate[0]))
}
