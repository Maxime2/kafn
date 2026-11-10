package kafn

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"

	tabulatedfunction "github.com/Maxime2/tabulated-function"

	"github.com/theothertomelliott/acyclic"
)

// Point is a point in Tabulated activation
type Point struct {
	X, Y float64
}

// Dump is a neural network dump
type Dump struct {
	Config   *Config
	Weights  [][][][]float64
	Synapses [][]Point
}

// ApplyWeights sets the weights from a four-dimensional slice
func (n *Neural) ApplyWeights(weights [][][][]float64) {
	for i, l := range n.Layers {
		if i >= len(weights) || weights[i] == nil {
			continue
		}
		if l.S != SynapseTypeTabulated {
			for j, neuron := range l.Neurons {
				if j >= len(weights[i]) || weights[i][j] == nil {
					continue
				}
				for k, in := range neuron.AnalyticIn {
					if k < len(weights[i][j]) {
						in.SetWeights(weights[i][j][k])
					}
				}
			}
		}
	}
}

// Weights returns all weights in sequence
func (n *Neural) Weights() [][][][]float64 {
	weights := make([][][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			weights[i] = make([][][]float64, len(l.Neurons))
			for j, neuron := range l.Neurons {
				weights[i][j] = make([][]float64, len(neuron.AnalyticIn))
				for k, in := range neuron.AnalyticIn {
					weights[i][j][k] = in.Weights
				}
			}
		}
	}
	return weights
}

func (n *Neural) ApplySynapses(points [][]Point) {
	current := 0
	for _, l := range n.Layers {
		if l.S == SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, s := range neuron.TabulatedIn {
					if current >= len(points) {
						current++
						continue
					}
					npoints := len(points[current])
					s.Clear()
					for i := 0; i < npoints; i++ {
						s.AddPoint(points[current][i].X, points[current][i].Y, n.Config.Epoch)
					}
					current++
				}
			}
		}
	}
}

func (n *Neural) Synapses() [][]Point {
	var synapses [][]Point
	for _, l := range n.Layers {
		if l.S == SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, s := range neuron.TabulatedIn {
					points := s.Len()
					acts := make([]Point, points)
					for i := 0; i < points; i++ {
						acts[i].X, acts[i].Y = s.GetPoint(i)
					}
					synapses = append(synapses, acts)
				}
			}
		}
	}
	return synapses

}

// Dump generates a network dump
func (n *Neural) Dump() *Dump {
	return &Dump{
		Config:   n.Config,
		Weights:  n.Weights(),
		Synapses: n.Synapses(),
	}
}

// FromDump restores a Neural from a dump
func FromDump(dump *Dump) *Neural {
	n := newNeuralRaw(dump.Config, false)
	n.ApplyWeights(dump.Weights)
	n.ApplySynapses(dump.Synapses)

	return n
}

// Marshal marshals to JSON from network
func (n *Neural) Marshal() ([]byte, error) {
	return json.Marshal(n.Dump())
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}

// Save saves network in readable JSON into the file specified
func (n *Neural) SaveReadable(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	acyclic.Fprint(f, n)
	return nil
}

type networkSaveData struct {
	Config   *Config
	Weights  [][]float64
	MinSums  []float64
	MaxSums  []float64
	TabDumps []*tabulatedfunction.Dump
}

// Save saves network into the file specified to be loaded later
func (n *Neural) Save(path string) error {
	if existingFile, err := os.Open(path); err == nil {
		var existingData networkSaveData
		dec := gob.NewDecoder(bufio.NewReader(existingFile))
		decodeErr := dec.Decode(&existingData)
		_ = existingFile.Close()
		if decodeErr == nil && existingData.Config != nil {
			var currentEpoch uint32
			if n.Config != nil {
				currentEpoch = n.Config.Epoch
			}
			if existingData.Config.Epoch >= currentEpoch {
				return nil
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	enc := gob.NewEncoder(w)

	data := networkSaveData{
		Config: n.Config,
	}
	for _, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, in := range neuron.AnalyticIn {
					data.Weights = append(data.Weights, in.Weights)
				}
				data.MinSums = append(data.MinSums, neuron.MinSum)
				data.MaxSums = append(data.MaxSums, neuron.MaxSum)
			}
		}
		if l.S == SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, s := range neuron.TabulatedIn {
					data.TabDumps = append(data.TabDumps, s.Dump())
				}
				data.MinSums = append(data.MinSums, neuron.MinSum)
				data.MaxSums = append(data.MaxSums, neuron.MaxSum)
			}
		}
	}

	if err := enc.Encode(&data); err != nil {
		return err
	}

	if err := w.Flush(); err != nil {
		return err
	}

	return nil
}

// Load retrieves network from the file specified created using Save method
func Load(path string) (*Neural, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	dec := gob.NewDecoder(r)

	var data networkSaveData
	if err := dec.Decode(&data); err != nil {
		return nil, err
	}

	n := newNeuralRaw(data.Config, false)

	wIdx := 0
	msIdx := 0
	tdIdx := 0

	for _, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, in := range neuron.AnalyticIn {
					if wIdx < len(data.Weights) {
						in.SetWeights(data.Weights[wIdx])
						wIdx++
					}
				}
				if msIdx < len(data.MinSums) {
					neuron.MinSum = data.MinSums[msIdx]
					neuron.MaxSum = data.MaxSums[msIdx]
					msIdx++
				}
			}
		}
		if l.S == SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, s := range neuron.TabulatedIn {
					if tdIdx < len(data.TabDumps) {
						s.FromDump(data.TabDumps[tdIdx])
						tdIdx++
					}
				}
				if msIdx < len(data.MinSums) {
					neuron.MinSum = data.MinSums[msIdx]
					neuron.MaxSum = data.MaxSums[msIdx]
					msIdx++
				}
			}
		}
	}
	return n, nil
}

// Save the network in DOT format for graphviz
func (n *Neural) Dot(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "digraph {\n")

	for l, lr := range n.Layers {
		for neuronIdx, nr := range lr.Neurons {
			for _, in := range nr.AnalyticIn {
				fmt.Fprintf(f, "\"%s\" -> \"L:%d N:%d\"[label=\"%v\"]\n",
					in.GetTag(), l, neuronIdx, in.WeightsString())
			}
			for _, in := range nr.TabulatedIn {
				fmt.Fprintf(f, "\"%s\" -> \"L:%d N:%d\"[label=\"%v\"]\n",
					in.GetTag(), l, neuronIdx, in.WeightsString())
			}
		}
	}

	fmt.Fprintf(f, "}\n")

	return nil
}

// Save the network in NET format
func (n *Neural) Net(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for l, lr := range n.Layers {
		fmt.Fprintf(f, "L: %d\n", l)
		for neuronIdx, nr := range lr.Neurons {
			fmt.Fprintf(f, "  N: %d;  Sum: %v; Ideal: %v\n", neuronIdx, nr.Sum, nr.Ideal)
			for _, in := range nr.AnalyticIn {
				fmt.Fprintf(f, " [%v %v]", in.GetIn(), in.GetOut())
			}
			for _, in := range nr.TabulatedIn {
				fmt.Fprintf(f, " [%v %v]", in.GetIn(), in.GetOut())
			}
			fmt.Fprintf(f, "\n")
		}
	}

	fmt.Fprintf(f, "\n")

	return nil
}
