// Reading and writing files are basic tasks needed for
// many Go programs. First we'll look at some examples of
// reading files.

package reader

import (
	//	"bufio"
	"fmt"
	//	"io"
	//"io/ioutil"
	"strconv"
	"testing"
)

// Reading files requires checking most calls for errors.
// This helper will streamline our error checks below.

func TestFile(t *testing.T) {
	fmt.Println("ExampleFile")
	var lblfilename = "/data/neocortexid/golang/minist/train-labels.idx1-ubyte"
	var imagefilename = "/data/neocortexid/golang/minist/train-images.idx3-ubyte"
	var labels = ReadMinistLabel(lblfilename)
	var images = ReadMinistImage(imagefilename)
	var size = len(labels)
	for i := 0; i < size; i++ {
		fmt.Println(strconv.Itoa(labels[i]))
	}
	fmt.Println(images[0].Values)
}
