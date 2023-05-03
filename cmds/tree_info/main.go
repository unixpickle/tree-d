package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/tree-d/treed"
)

func main() {
	flag.Parse()

	args := flag.Args()
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "Usage: tree_info [flags] <input.bin>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}
	inputPath := args[0]

	log.Println("Loading tree...")
	tree, err := treed.Load(inputPath, treed.ReadBoundedSolidTree)
	essentials.Must(err)

	fmt.Println("Number of leaves:", tree.Tree.NumLeaves())
}
