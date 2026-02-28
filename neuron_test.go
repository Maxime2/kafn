package kafn

import (
	"testing"

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
