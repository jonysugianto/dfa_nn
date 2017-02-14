package reader

import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"

	"github.com/jonysugianto/mathlib/matrix"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func ConvLabelToTargetVector(l int) *matrix.Vector {
	var ret = matrix.CreateVector(10)
	ret.Values[l] = 1
	return ret
}

func ReadMinistLabel(filename string) []int {
	f, err := os.Open(filename)
	check(err)

	var buf4bytes = make([]byte, 4)
	_, err = f.Read(buf4bytes)
	check(err)

	var v = binary.BigEndian.Uint32(buf4bytes)
	fmt.Print("magicnumber ")
	fmt.Println(strconv.Itoa(int(v)))

	_, err = f.Read(buf4bytes)
	check(err)

	v = binary.BigEndian.Uint32(buf4bytes)
	fmt.Print("number images ")
	fmt.Println(strconv.Itoa(int(v)))

	var b = make([]byte, 1)
	var ret []int
	var size = int(v)
	for i := 0; i < size; i++ {
		_, err = f.Read(b)
		check(err)
		ret = append(ret, int(b[0]))
	}
	return ret
}

func ReadMinistImage(filename string) []*matrix.Vector {
	f, err := os.Open(filename)
	check(err)

	var buf4bytes = make([]byte, 4)
	_, err = f.Read(buf4bytes)
	check(err)

	var v = binary.BigEndian.Uint32(buf4bytes)
	fmt.Print("magicnumber ")
	fmt.Println(strconv.Itoa(int(v)))

	_, err = f.Read(buf4bytes)
	check(err)

	v = binary.BigEndian.Uint32(buf4bytes)
	fmt.Print("number images ")
	fmt.Println(strconv.Itoa(int(v)))
	var size = int(v)

	_, err = f.Read(buf4bytes)
	check(err)

	v = binary.BigEndian.Uint32(buf4bytes)
	fmt.Print("row ")
	fmt.Println(strconv.Itoa(int(v)))

	_, err = f.Read(buf4bytes)
	check(err)

	v = binary.BigEndian.Uint32(buf4bytes)
	fmt.Print("col ")
	fmt.Println(strconv.Itoa(int(v)))

	var b = make([]byte, 1)
	var ret []*matrix.Vector
	for i := 0; i < size; i++ {
		var img []float64
		for j := 0; j < 784; j++ {
			_, err = f.Read(b)
			check(err)
			img = append(img, float64(int(b[0]))/255.0)
		}
		var v = matrix.CreateVector(len(img))
		v.Copy(img)
		ret = append(ret, v)
	}
	return ret
}
