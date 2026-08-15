package kafn

import (
	"encoding/binary"
	"errors"
	"math"
)

type Deepfloat64 float64

func DF(f float64) Deepfloat64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0.0
	}
	return Deepfloat64(f)
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
	return float64(a)
}

// GobEncode implements the gob.GobEncoder interface.
func (d *Deepfloat64) GobEncode() ([]byte, error) {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], math.Float64bits(float64(*d)))
	return buf[:], nil
}

// GobDecode implements the gob.GobDecoder interface.
func (d *Deepfloat64) GobDecode(data []byte) error {
	if len(data) < 8 {
		return errors.New("truncated float64 data")
	}
	*d = Deepfloat64(math.Float64frombits(binary.BigEndian.Uint64(data)))
	return nil
}
