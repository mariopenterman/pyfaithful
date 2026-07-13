#include <cstring>
#include <fstream>
#include <iostream>
#include <streambuf>
#include "ASTree.h"
#include "ASTNode.h"

#ifdef WIN32
#  define PATHSEP '\\'
#else
#  define PATHSEP '/'
#endif

/* Forwarding stream buffer that counts emitted newlines into g_curLine, so the
   layout engine always knows the current output line. Every byte written to the
   decompiler output (header comment included) passes through here. */
class LineCountStreambuf : public std::streambuf {
public:
    explicit LineCountStreambuf(std::streambuf* dst) : m_dst(dst) {}
protected:
    int overflow(int ch) override {
        if (ch == EOF)
            return ch;
        if (ch == '\n') {
            ++g_curLine;
            g_curCol = 0;
        } else {
            ++g_curCol;
        }
        return m_dst->sputc(static_cast<char>(ch));
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        for (std::streamsize i = 0; i < n; ++i) {
            if (s[i] == '\n') {
                ++g_curLine;
                g_curCol = 0;
            } else {
                ++g_curCol;
            }
        }
        return m_dst->sputn(s, n);
    }
    int sync() override { return m_dst->pubsync(); }
private:
    std::streambuf* m_dst;
};

int main(int argc, char* argv[])
{
    const char* infile = nullptr;
    bool marshalled = false;
    const char* version = nullptr;
    std::ostream* pyc_output = &std::cout;
    std::ofstream out_file;

    for (int arg = 1; arg < argc; ++arg) {
        if (strcmp(argv[arg], "-o") == 0) {
            if (arg + 1 < argc) {
                const char* filename = argv[++arg];
                out_file.open(filename, std::ios_base::out);
                if (out_file.fail()) {
                    fprintf(stderr, "Error opening file '%s' for writing\n",
                            filename);
                    return 1;
                }
                pyc_output = &out_file;
            } else {
                fputs("Option '-o' requires a filename\n", stderr);
                return 1;
            }
        } else if (strcmp(argv[arg], "-c") == 0) {
            marshalled = true;
        } else if (strcmp(argv[arg], "-v") == 0) {
            if (arg + 1 < argc) {
                version = argv[++arg];
            } else {
                fputs("Option '-v' requires a version\n", stderr);
                return 1;
            }
        } else if (strcmp(argv[arg], "--help") == 0 || strcmp(argv[arg], "-h") == 0) {
            fprintf(stderr, "Usage:  %s [options] input.pyc\n\n", argv[0]);
            fputs("Options:\n", stderr);
            fputs("  -o <filename>  Write output to <filename> (default: stdout)\n", stderr);
            fputs("  -c             Specify loading a compiled code object. Requires the version to be set\n", stderr);
            fputs("  -v <x.y>       Specify a Python version for loading a compiled code object\n", stderr);
            fputs("  --help         Show this help text and then exit\n", stderr);
            return 0;
        } else {
            infile = argv[arg];
        }
    }

    if (!infile) {
        fputs("No input file specified\n", stderr);
        return 1;
    }

    PycModule mod;
    if (!marshalled) {
        try {
            mod.loadFromFile(infile);
        } catch (std::exception& ex) {
            fprintf(stderr, "Error loading file %s: %s\n", infile, ex.what());
            return 1;
        }
    } else {
        if (!version) {
            fputs("Opening raw code objects requires a version to be specified\n", stderr);
            return 1;
        }
        std::string s(version);
        auto dot = s.find('.');
        if (dot == std::string::npos || dot == s.size()-1) {
            fputs("Unable to parse version string (use the format x.y)\n", stderr);
            return 1;
        }
        int major = std::stoi(s.substr(0, dot));
        int minor = std::stoi(s.substr(dot+1, s.size()));
        mod.loadFromMarshalledFile(infile, major, minor);
    }

    if (!mod.isValid()) {
        fprintf(stderr, "Could not load file %s\n", infile);
        return 1;
    }
    /* Route all output through the newline counter so g_curLine tracks the
       true output line. No header comment is emitted: the layout engine places
       every statement on its original source line, and a 2-line header would
       occupy output lines 1-2, making any construct on source lines 1-2
       impossible to align (module docstring, early imports). Dropping it lets
       the module body start at line 1 and match the original line table. */
    LineCountStreambuf lcbuf(pyc_output->rdbuf());
    std::ostream counted(&lcbuf);
    g_curLine = 1;
    try {
        decompyle(mod.code(), &mod, counted);
    } catch (std::exception& ex) {
        fprintf(stderr, "Error decompyling %s: %s\n", infile, ex.what());
        return 1;
    }

    return 0;
}
