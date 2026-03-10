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
	// Register concrete types for interfaces that will be encoded with gob.
	// The `Trapolation` field in the `Config` struct is an interface, and this
	// ensures that gob knows how to serialize/deserialize its concrete implementations.
	// If other types can be assigned to `Trapolation`, they must be registered too.
	gob.Register(tabulatedfunction.TrapolationLinear)
}

// Dump is a neural network dump
type Dump struct {
	Config      *Config
	Weights     [][][][]Deepfloat64
	Activations [][]Point
	Synapses    [][]Point
}

// ApplyWeights sets the weights from a four-dimensional slice
func (n *Neural) ApplyWeights(weights [][][][]Deepfloat64) {
	for i, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			for j := range l.Neurons {
				for k := range l.Neurons[j].In {
					n.Layers[i].Neurons[j].In[k].SetWeights(weights[i][j][k])
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
			for j, n := range l.Neurons {
				weights[i][j] = make([][]Deepfloat64, len(n.In))
				for k, in := range n.In {
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
			for _, n := range l.Neurons {
				for _, s := range n.In {
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
	enc.Encode(n.Config)

	// Store Weights
	for _, l := range n.Layers {
		if l.S != SynapseTypeTabulated {
			for _, n := range l.Neurons {
				for _, in := range n.In {
					enc.Encode(in.GetWeights())
				}
			}
		}
	}

	// Store Tabulated Synapses
	for _, l := range n.Layers {
		if l.S == SynapseTypeTabulated {
			for _, n := range l.Neurons {
				for _, s := range n.In {
					if tab, ok := s.(*SynapseTabulated); ok {
						enc.Encode(tab.Dump())
					}
				}
			}
		}
	}

	w.Flush()

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
			for _, n := range l.Neurons {
				for _, in := range n.In {
					var w []Deepfloat64
					if err := dec.Decode(&w); err != nil {
						return nil, err
					}
					in.SetWeights(w)
				}
			}
		}
	}

	// Restore Tabulated Synapses
	for _, l := range n.Layers {
		if l.S == SynapseTypeTabulated {
			for _, n := range l.Neurons {
				for _, s := range n.In {
					if tab, ok := s.(*SynapseTabulated); ok {
						var dump tabulatedfunction.Dump
						if err := dec.Decode(&dump); err != nil {
							return nil, err
						}
						tab.FromDump(&dump)
					}
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
		for n, nr := range lr.Neurons {
			for _, in := range nr.In {
				fmt.Fprintf(f, "\"%s\" -> \"L:%d N:%d\"[label=\"%v\"]\n",
					in.GetTag(), l, n, in.WeightsString())
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
		for n, nr := range lr.Neurons {
			fmt.Fprintf(f, "  N: %d;  Sum: %v; Ideal: %v\n", n, nr.Sum, nr.Ideal)
			for _, in := range nr.In {
				fmt.Fprintf(f, " [%v %v]", in.GetIn(), in.GetOut())
			}
			fmt.Fprintf(f, "\n")
		}
	}

	fmt.Fprintf(f, "\n")

	return nil
}
