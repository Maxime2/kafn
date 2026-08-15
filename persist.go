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
	X, Y Deepfloat64
}

func init() {
	// tabulatedfunction.Trapolation is a concrete type (int alias),
	// so gob registration is not required for serialization.
}

// Dump is a neural network dump
type Dump struct {
	Config   *Config
	Weights  [][][][]Deepfloat64
	Synapses [][]Point
}

// ApplyWeights sets the weights from a four-dimensional slice
func (n *Neural) ApplyWeights(weights [][][][]Deepfloat64) {
	for i, l := range n.Layers {
		if i >= len(weights) || weights[i] == nil {
			continue
		}
		if l.S != SynapseTypeTabulated {
			for j, neuron := range l.Neurons {
				if j >= len(weights[i]) || weights[i][j] == nil {
					continue
				}
				for k, in := range neuron.In {
					if k < len(weights[i][j]) {
						in.SetWeights(weights[i][j][k])
					}
				}
			}
		}
	}
}

// Weights returns all weights in sequence
func (n *Neural) Weights() [][][][]Deepfloat64 {
	weights := make([][][][]Deepfloat64, len(n.Layers))
	for i, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			weights[i] = make([][][]Deepfloat64, len(l.Neurons))
			for j, neuron := range l.Neurons {
				weights[i][j] = make([][]Deepfloat64, len(neuron.In))
				for k, in := range neuron.In {
					weights[i][j][k] = in.GetWeights()
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
				for _, s := range neuron.In {
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
				for _, s := range neuron.In {
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
	n := NewNeural(dump.Config)
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

// Save saves network into the file specified to be loaded later
func (n *Neural) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	enc := gob.NewEncoder(w)

	// Store Config
	if err := enc.Encode(n.Config); err != nil {
		return err
	}

	// Store Weights
	for _, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, in := range neuron.In {
					if err := enc.Encode(in.GetWeights()); err != nil {
						return err
					}
				}
				if err := enc.Encode(&neuron.MinSum); err != nil {
					return err
				}
				if err := enc.Encode(&neuron.MaxSum); err != nil {
					return err
				}
			}
		}
	}

	// Store Tabulated Synapses
	for _, l := range n.Layers {
		if l.S == SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, s := range neuron.In {
					if tab, ok := s.(*SynapseTabulated); ok {
						if err := enc.Encode(tab.Dump()); err != nil {
							return err
						}
					}
				}
				if err := enc.Encode(&neuron.MinSum); err != nil {
					return err
				}
				if err := enc.Encode(&neuron.MaxSum); err != nil {
					return err
				}
			}
		}
	}

	if err := w.Flush(); err != nil {
		return err
	}

	return nil
}

// Load retrieves network from the file specified created using Save method
func Load(path string) (*Neural, error) {
	var config Config

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	dec := gob.NewDecoder(r)

	// Restore config
	if err := dec.Decode(&config); err != nil {
		return nil, err
	}

	n := NewNeural(&config)

	// Restore Weights for Analytic Synapses
	// Restore Weights
	for _, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, in := range neuron.In {
					var w []Deepfloat64
					if err := dec.Decode(&w); err != nil {
						return nil, err
					}
					in.SetWeights(w)
				}
				if err := dec.Decode(&neuron.MinSum); err != nil {
					return nil, err
				}
				if err := dec.Decode(&neuron.MaxSum); err != nil {
					return nil, err
				}
			}
		}
	}

	// Restore Tabulated Synapses
	for _, l := range n.Layers {
		if l.S == SynapseTypeTabulated {
			for _, neuron := range l.Neurons {
				for _, s := range neuron.In {
					if tab, ok := s.(*SynapseTabulated); ok {
						var dump tabulatedfunction.Dump
						if err := dec.Decode(&dump); err != nil {
							return nil, err
						}
						tab.FromDump(&dump)
					}
				}
				if err := dec.Decode(&neuron.MinSum); err != nil {
					return nil, err
				}
				if err := dec.Decode(&neuron.MaxSum); err != nil {
					return nil, err
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
			for _, in := range nr.In {
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
			for _, in := range nr.In {
				fmt.Fprintf(f, " [%v %v]", in.GetIn(), in.GetOut())
			}
			fmt.Fprintf(f, "\n")
		}
	}

	fmt.Fprintf(f, "\n")

	return nil
}
