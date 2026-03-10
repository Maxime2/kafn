package kafn

import (
	"bytes"
	"encoding/binary"
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
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.BigEndian, float64(*d))
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode implements the gob.GobDecoder interface.
func (d *Deepfloat64) GobDecode(data []byte) error {
	buf := bytes.NewReader(data)
	var f float64
	err := binary.Read(buf, binary.BigEndian, &f)
	if err != nil {
		return err
	}
	*d = Deepfloat64(f)
	return nil
}
