package collective

import (
	"math"
	"math/rand/v2"
	"testing"
)

func randomVectors(n, size int, seed uint64) [][]float64 {
	rng := rand.New(rand.NewPCG(seed, 0))
	vs := make([][]float64, n)
	for r := range vs {
		vs[r] = make([]float64, size)
		for i := range vs[r] {
			vs[r][i] = rng.Float64()
		}
	}
	return vs
}

func expectedSum(vs [][]float64) []float64 {
	sum := make([]float64, len(vs[0]))
	for _, v := range vs {
		add(sum, v)
	}
	return sum
}

func TestAllReduceIsCorrect(t *testing.T) {
	ops := map[string]AllReduce{"central": Central, "ring": Ring}
	for name, op := range ops {
		for _, n := range []int{1, 2, 3, 4, 7, 8} {
			for _, size := range []int{1, 5, 64, 1000} {
				vs := randomVectors(n, size, uint64(n*size))
				want := expectedSum(vs)
				Run(NewComm(n), op, vs)
				for r := range n {
					for i := range want {
						if math.Abs(vs[r][i]-want[i]) > 1e-9 {
							t.Fatalf("%s n=%d size=%d: rank %d pos %d = %v, want %v",
								name, n, size, r, i, vs[r][i], want[i])
						}
					}
				}
			}
		}
	}
}

// El resultado tiene que ser idéntico bit a bit en todos los ranks: si no,
// las réplicas del modelo divergen. (Ver la nota sobre MPI en Patarasuk y Yuan.)
func TestAllRanksAgreeExactly(t *testing.T) {
	for _, op := range []AllReduce{Central, Ring} {
		vs := randomVectors(6, 999, 42)
		Run(NewComm(6), op, vs)
		for r := 1; r < 6; r++ {
			for i := range vs[0] {
				if vs[r][i] != vs[0][i] {
					t.Fatalf("rank %d difiere del rank 0 en pos %d", r, i)
				}
			}
		}
	}
}

func TestRingIsBandwidthOptimal(t *testing.T) {
	const size = 1200 // divisible por todos los n de abajo
	for _, n := range []int{2, 3, 4, 6, 8} {
		c := NewComm(n)
		Run(c, Ring, randomVectors(n, size, 1))
		want := int64(2 * (n - 1) * size / n)
		for r := range n {
			if got := c.SentBy(r); got != want {
				t.Errorf("ring n=%d rank %d envió %d floats, want 2(n-1)/n·S = %d", n, r, got, want)
			}
		}
	}
}

func TestCentralBottleneckGrowsWithN(t *testing.T) {
	const size = 1000
	for _, n := range []int{2, 4, 8} {
		c := NewComm(n)
		Run(c, Central, randomVectors(n, size, 1))
		if got, want := c.SentBy(0), int64((n-1)*size); got != want {
			t.Errorf("central n=%d: rank 0 envió %d, want (n-1)·S = %d", n, got, want)
		}
	}
}
