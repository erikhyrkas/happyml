#include <iostream>
#include "types/quarter_float.hpp"


using namespace microml;

int main() {
    // TODO: read from input stream (this can be used to handle API requests or files or whatever later.)
    // I'll need to make a lexer/parser and think through the details more.
    // I want to make this similar to sql, in that the underlying details of how
    // are abstracted away, but I'll try to find ways to give hints to allow people control

    // syntax might be something like:
    // set output <human/machine>

    // create <data set type> dataset <name> from <location> [with <format>]
    // add rows to dataset <name> using delimited data:
    // 1, 2, 3, 4
    // <empty line to denote end of data>

    // create [<adjective>*] <model type> model <model name> [<model version>] using <data set name>
    // tune <model name> [<model version>] [as [<model name>] [<model version>]] using <data set name>
    // retrain <model name> [<model version>] [as [<model name>] [<model version>]] using <data set name>

    // predict using <model name> [<model version] given <input>
    //               or
    // infer using <model name> [<model version] given <input>

    return 0;
}
