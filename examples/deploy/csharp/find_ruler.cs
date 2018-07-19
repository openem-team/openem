using System;

class Program {
  static int Main(string[] args) {

    // Check input arguments.
    if(args.Length < 2) {
      Console.WriteLine("Expected at least two arguments:");
      Console.WriteLine("  Path to protobuf file containing model");
      Console.WriteLine("  Paths to one or more image files");
      return -1;
    }

    return 0;
  }
}

