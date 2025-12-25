package kafn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_RestoreFromDump(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	dump := n.Dump()
	new := FromDump(dump)

	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Predict([]Deepfloat64{0}), new.Predict([]Deepfloat64{0}))
}

func Test_Marshal(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	dump, err := n.Marshal()
	assert.Nil(t, err)

	new, err := Unmarshal(dump)
	assert.Nil(t, err)

	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Predict([]Deepfloat64{0}), new.Predict([]Deepfloat64{0}))
}
