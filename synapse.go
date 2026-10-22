package kafn

import (
	"fmt"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

type SynapseType int

const (
	// SynapseAnalytic is an analytical function
	SynapseTypeAnalytic SynapseType = 0
	// SynapseTabulated is a tabulated function
	SynapseTypeTabulated SynapseType = 1
)


type SynapseTabulated struct {
	direct  *tabulatedfunction.TabulatedFunction
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
	// Initialize with 100 random points.
	for i := 0; i < 100; i++ {
		x := Rand.Float64() * float64(c.sum_of_weights)
		y := 0.3 + Rand.Float64()*0.4
		syn.AddPoint(DF(x), DF(y), 0)
	}
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
	s.Out = DF(s.direct.F(float64(s.In)))
}

func (s *SynapseTabulated) RefireT(trapolation tabulatedfunction.Trapolation) {
	s.Out = DF(s.direct.Trapolate(float64(s.In), trapolation))
}

func (s *SynapseTabulated) Fire(value Deepfloat64) {
	s.In = value
	s.Refire()
}

func (s *SynapseTabulated) FireT(value Deepfloat64, trapolation tabulatedfunction.Trapolation) {
	s.In = value
	s.RefireT(trapolation)
}

func (s *SynapseTabulated) SetTag(tag string) {
	s.Tag = tag
}

func (s *SynapseTabulated) GetTag() string {
	return s.Tag
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

func (s *SynapseTabulated) GetUp() *Neuron {
	return s.Up
}

func (s *SynapseTabulated) AddPoint(x, y Deepfloat64, it uint32) {
	valY := y
	if valY < 0 {
		valY = DF(0)
	}
	s.direct.AddPoint(float64(x), float64(valY), it)
}

func (s *SynapseTabulated) Trapolate(x Deepfloat64, trapolation tabulatedfunction.Trapolation) Deepfloat64 {
	return DF(s.direct.Trapolate(float64(x), trapolation))
}

// GetPoint() returns n-th point in Tabulated activation
func (s *SynapseTabulated) GetPoint(i int) (Deepfloat64, Deepfloat64) {
	return DF(s.direct.X[i]), DF(s.direct.Y[i])
}

func (s *SynapseTabulated) DrawPS(path string) {
	s.direct.DrawPS(path)
}

func (s *SynapseTabulated) Smooth() {
	s.direct.Smooth()
}

// Dump returns a serializable dump of the underlying tabulated function.
func (s *SynapseTabulated) Dump() *tabulatedfunction.Dump {
	return s.direct.Dump()
}

// FromDump restores the underlying tabulated function from a dump.
func (s *SynapseTabulated) FromDump(d *tabulatedfunction.Dump) {
	s.direct.FromDump(d)
}

func (s *SynapseTabulated) Polinate(m *SynapseTabulated) {
	for i := 0; i < m.Len(); i++ {
		X, Y := m.GetPoint(i)
		s.AddPoint(X, Y, 0 /* epoch */)
	}
	for i := 0; i < s.Len(); i++ {
		X, Y := s.GetPoint(i)
		m.AddPoint(X, Y, 0 /* epoch */)
	}
}

func (s *SynapseTabulated) Expand(n int) {
	s.direct.Expand(n)
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

	copy(weights, init_weights)

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

func (s *SynapseAnalytic) SetTag(tag string) {
	s.Tag = tag
}
func (s *SynapseAnalytic) GetTag() string {
	return s.Tag
}

func (s *SynapseAnalytic) SetWeight(k int, weight Deepfloat64) {
	s.Weights[k] = weight
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
