package kafn

import (
	"fmt"
	"math"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

type SynapseType int

const (
	// SynapseAnalytic is an analytical function
	SynapseTypeAnalytic SynapseType = 0
	// SynapseTabulated is a tabulated function
	SynapseTypeTabulated SynapseType = 1
)

// Synapse is an edge between neurons
type Synapse interface {
	Refire()
	Fire(Deepfloat64)
	FireT(Deepfloat64, tabulatedfunction.Trapolation)
	FireDerivative() Deepfloat64
	SetTag(string)
	GetTag() string
	SetWeight(int, Deepfloat64)
	GetGradient(Deepfloat64, int) Deepfloat64
	GetWeight(int) Deepfloat64
	String() string
	WeightsString() string
	GetIn() Deepfloat64
	GetOut() Deepfloat64
	Len() int
	SetWeights([]Deepfloat64)
	GetWeights() []Deepfloat64
	GetUp() *Neuron
	Epoch(uint32)
	AddPoint(x, y Deepfloat64, it uint32)
	Trapolate(x Deepfloat64, trapolation tabulatedfunction.Trapolation) Deepfloat64
	GetPoint(i int) (Deepfloat64, Deepfloat64)
	DrawPS(path string)
	Smooth()
	Polinate(Synapse)
	Clear()
}

type SynapseTabulated struct {
	direct  *tabulatedfunction.TabulatedFunction
	changed bool
	Up      *Neuron
	In, Out Deepfloat64
	Tag     string
}

// NewSynapseTabulated returns a tabulated function synapse preset with specific tag
func NewSynapseTabulated(c *Config, up *Neuron, tag string) *SynapseTabulated {
	direct := tabulatedfunction.New()
	direct.SetOrder(1)
	direct.SetTrapolation(c.Trapolation)
	syn := &SynapseTabulated{
		direct: direct,
		In:     DF(0),
		Out:    DF(0),
		Tag:    tag,
		Up:     up,
	}
	syn.direct.LoadConstant(DF(0.5), DF(0), DF(1))
	return syn
}

func (s *SynapseTabulated) String() string {
	return fmt.Sprintf("Tag: %v; In: %v; Out: %v; %v",
		s.Tag, s.In, s.Out, s.direct.String())
}

func (s *SynapseTabulated) WeightsString() string {
	return "[tabulated]"
}

func (s *SynapseTabulated) Refire() {
	s.Out = Copy(s.direct.F(s.In))
}

func (s *SynapseTabulated) RefireT(trapolation tabulatedfunction.Trapolation) {
	s.Out = Copy(s.direct.Trapolate(s.In, trapolation))
}

func (s *SynapseTabulated) Fire(value Deepfloat64) {
	s.In = value
	s.Refire()
}

func (s *SynapseTabulated) FireT(value Deepfloat64, trapolation tabulatedfunction.Trapolation) {
	s.In = value
	s.RefireT(trapolation)
}

func (s *SynapseTabulated) FireDerivative() Deepfloat64 {
	return DF(0)
}

func (s *SynapseTabulated) SetTag(tag string) {
	s.Tag = tag
}

func (s *SynapseTabulated) GetTag() string {
	return s.Tag
}

func (s *SynapseTabulated) SetWeight(k int, weight Deepfloat64) {

}

func (s *SynapseTabulated) GetGradient(D_E_x Deepfloat64, k int) Deepfloat64 {
	return Mul(D_E_x, DF(math.Pow(Float64(s.In), float64(k))))
}

func (s *SynapseTabulated) GetWeight(k int) Deepfloat64 {
	return DF(math.NaN())
}

func (s *SynapseTabulated) Epoch(epoch uint32) {
	s.direct.Epoch(epoch)
}

func (s *SynapseTabulated) Clear() {
	s.direct.Clear()
}

func (s *SynapseTabulated) GetIn() Deepfloat64 {
	return s.In
}

func (s *SynapseTabulated) GetOut() Deepfloat64 {
	return s.Out
}

func (s *SynapseTabulated) Len() int {
	return s.direct.GetNdots()
}

func (s *SynapseTabulated) SetWeights(w []Deepfloat64) {}
func (s *SynapseTabulated) GetWeights() []Deepfloat64 {
	return []Deepfloat64{}
}

func (s *SynapseTabulated) GetUp() *Neuron {
	return s.Up
}

func (s *SynapseTabulated) AddPoint(x, y Deepfloat64, it uint32) {
	valY := y
	if valY.Sign() < 0 {
		valY = DF(0)
	}
	s.direct.AddPoint(x, valY, it)
	s.changed = true
}

func (s *SynapseTabulated) Trapolate(x Deepfloat64, trapolation tabulatedfunction.Trapolation) Deepfloat64 {
	return Copy(s.direct.Trapolate(x, trapolation))
}

// GetPoint() returns n-th point in Tabulated activation
func (s *SynapseTabulated) GetPoint(i int) (Deepfloat64, Deepfloat64) {
	return s.direct.P[i].X, s.direct.P[i].Y
}

func (s *SynapseTabulated) DrawPS(path string) {
	s.direct.DrawPS(path)
}

func (s *SynapseTabulated) Smooth() {
	s.direct.Smooth()
}

func (s *SynapseTabulated) Polinate(m Synapse) {
	for i := range m.Len() {
		X, Y := m.GetPoint(i)
		s.AddPoint(X, Y, 0 /* epoch */)
	}
	for i := range s.Len() {
		X, Y := s.GetPoint(i)
		m.AddPoint(X, Y, 0 /* epoch */)
	}
}

type SynapseAnalytic struct {
	Weights []Deepfloat64
	Up      *Neuron
	In, Out Deepfloat64
	Tag     string
}

// NewSynapseAnalytic returns a synapse with the weigths preset with specified initializer
// and marked with specified tag
func NewSynapseAnalytic(up *Neuron, degree int, init_weights []Deepfloat64, tag string) *SynapseAnalytic {
	var weights = make([]Deepfloat64, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = init_weights[i]
	}
	return &SynapseAnalytic{
		Weights: weights,
		In:      DF(0),
		Out:     DF(0),
		Tag:     tag,
		Up:      up,
	}
}

func (s *SynapseAnalytic) String() string {
	return fmt.Sprintf("Tag: %v; In: %v; Out: %v; Weights: %v",
		s.Tag, s.In, s.Out, s.Weights)
}

func (s *SynapseAnalytic) WeightsString() string {
	return fmt.Sprintf("%v", s.Weights)
}

func (s *SynapseAnalytic) Refire() {
	mul := DF(1)
	s.Out = DF(0)
	for k := 0; k < len(s.Weights); k++ {
		s.Out = Add(s.Out, Mul(s.Weights[k], mul))
		mul = Mul(mul, s.In)
	}
}

func (s *SynapseAnalytic) Fire(value Deepfloat64) {
	s.In = value
	s.Refire()
}

func (s *SynapseAnalytic) FireT(value Deepfloat64, trapolation tabulatedfunction.Trapolation) {
	s.In = value
	s.Refire()
}

func (s *SynapseAnalytic) FireDerivative() Deepfloat64 {
	mul := DF(1)
	res := DF(0)
	for k := 1; k < len(s.Weights); k++ {
		res = Add(res, Mul(DF(float64(k)), Mul(mul, s.Weights[k])))
		mul = Mul(mul, s.In)
	}
	return res
}

func (s *SynapseAnalytic) SetTag(tag string) {
	s.Tag = tag
}
func (s *SynapseAnalytic) GetTag() string {
	return s.Tag
}

func (s *SynapseAnalytic) SetWeight(k int, weight Deepfloat64) {
	s.Weights[k] = weight
}

func (s *SynapseAnalytic) GetGradient(D_E_x Deepfloat64, k int) Deepfloat64 {
	return Mul(D_E_x, DF(math.Pow(Float64(s.In), float64(k))))
}

func (s *SynapseAnalytic) GetWeight(k int) Deepfloat64 {
	return s.Weights[k]
}

func (s *SynapseAnalytic) GetIn() Deepfloat64 {
	return s.In
}

func (s *SynapseAnalytic) GetOut() Deepfloat64 {
	return s.Out
}

func (s *SynapseAnalytic) Len() int {
	return len(s.Weights)
}

func (s *SynapseAnalytic) SetWeights(w []Deepfloat64) {
	s.Weights = w
}

func (s *SynapseAnalytic) GetWeights() []Deepfloat64 {
	return s.Weights
}

func (s *SynapseAnalytic) GetUp() *Neuron {
	return s.Up
}

func (s *SynapseAnalytic) Epoch(uint32) {}

func (s *SynapseAnalytic) Clear() {}

func (s *SynapseAnalytic) AddPoint(x, y Deepfloat64, it uint32) {}

func (s *SynapseAnalytic) Trapolate(x Deepfloat64, trapolation tabulatedfunction.Trapolation) Deepfloat64 {
	return DF(math.NaN())
}

// GetPoint() returns (0,0,1)
func (s *SynapseAnalytic) GetPoint(i int) (Deepfloat64, Deepfloat64) { return DF(0), DF(0) }

func (s *SynapseAnalytic) DrawPS(path string) {}

func (s *SynapseAnalytic) Smooth() {}

func (s *SynapseAnalytic) Polinate(m Synapse) {}
