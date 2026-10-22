package kafn

import (
	"math"
	"testing"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
	"github.com/stretchr/testify/assert"
)

func Test_Synapse_Fire(t *testing.T) {
	s := &SynapseAnalytic{
		Weights: []Deepfloat64{DF(277.9848903746018),
			DF(-46.494014333557395),
			DF(-52.90141723171177),
			DF(-53.09001803405063),
			DF(-53.10138885017573),
			DF(-53.10279737262996),
			DF(-53.10308684399308),
			DF(-53.10317433993435)},
		In:  DF(7.9556310898783e-14),
		Out: DF(0),
		Tag: "test",
	}
	s.Fire(DF(7.9556310898783e-14))
	assert.InEpsilon(t, 277.984890374598, Float64(s.Out), 0.0000001)
}

func Test_SynapseAnalytic_Full(t *testing.T) {
	parentNeuron := NewNeuron()
	syn := NewSynapseAnalytic(parentNeuron, 1, []Deepfloat64{DF(1.5), DF(2.0)}, "test-analytic")

	assert.Equal(t, "test-analytic", syn.GetTag())
	syn.SetTag("new-tag")
	assert.Equal(t, "new-tag", syn.GetTag())

	assert.Equal(t, parentNeuron, syn.GetUp())
	assert.Equal(t, 2, syn.Len())

	// Weights
	syn.SetWeight(0, DF(3.0))
	assert.Equal(t, DF(3.0), syn.GetWeight(0))
	syn.SetWeights([]Deepfloat64{DF(1.0), DF(2.0)})
	assert.Equal(t, []Deepfloat64{DF(1.0), DF(2.0)}, syn.GetWeights())

	// Fire and FireT
	syn.Fire(DF(2.0)) // 1.0 + 2.0*2.0 = 5.0
	assert.Equal(t, DF(2.0), syn.GetIn())
	assert.Equal(t, DF(5.0), syn.GetOut())
	syn.FireT(DF(3.0), tabulatedfunction.TrapolationLinear)
	assert.Equal(t, DF(7.0), syn.GetOut())

	// String methods and dummy operations
	assert.NotEmpty(t, syn.String())
	assert.NotEmpty(t, syn.WeightsString())
	assert.Equal(t, DF(0), syn.Trapolate(DF(1), tabulatedfunction.TrapolationLinear))
	x, y := syn.GetPoint(0)
	assert.Equal(t, DF(0), x)
	assert.Equal(t, DF(0), y)

	assert.NotPanics(t, func() {
		syn.Epoch(1)
		syn.Clear()
		syn.AddPoint(DF(0), DF(0), 0)
		syn.DrawPS("")
		syn.Smooth()
		syn.Polinate(syn)
		syn.Expand(1)
	})
}

func Test_SynapseTabulated_Full(t *testing.T) {
	parentNeuron := NewNeuron()
	config := &Config{Trapolation: tabulatedfunction.TrapolationLinear, sum_of_weights: DF(1.0)}
	syn := NewSynapseTabulated(config, parentNeuron, "test-tab")

	assert.Equal(t, "test-tab", syn.GetTag())
	syn.SetTag("tab-tag")
	assert.Equal(t, "tab-tag", syn.GetTag())
	assert.Equal(t, parentNeuron, syn.GetUp())

	// Negative Y clamp in AddPoint
	syn.AddPoint(DF(0.5), DF(-1.0), 0)
	assert.NotEmpty(t, syn.String())
	assert.Equal(t, "[tabulated]", syn.WeightsString())

	// Weights and dummy ops
	syn.SetWeight(0, DF(1.0))
	assert.Equal(t, DF(0), syn.GetWeight(0))
	syn.SetWeights([]Deepfloat64{})
	assert.Empty(t, syn.GetWeights())

	// Fire and FireT
	syn.Fire(DF(0.5))
	syn.FireT(DF(0.5), tabulatedfunction.TrapolationLinear)
	assert.NotPanics(t, func() {
		_ = syn.Trapolate(DF(0.5), tabulatedfunction.TrapolationLinear)
		syn.Epoch(1)
		syn.Smooth()
	})
}

func Test_Neuron_FireT_And_NanTolerance(t *testing.T) {
	n := NewNeuron()
	// Synapse returning NaN should be skipped during sum calculation with a log warning
	badSynapse := &SynapseAnalytic{Out: Deepfloat64(math.NaN())}
	goodSynapse := &SynapseAnalytic{Out: DF(5.0)}
	n.In = []Synapse{badSynapse, goodSynapse}

	n.fireT(tabulatedfunction.TrapolationLinear)
	assert.Equal(t, DF(5.0), n.Sum)
}
