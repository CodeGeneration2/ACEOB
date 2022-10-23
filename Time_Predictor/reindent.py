'\nReindent files.\n'
from __future__ import print_function
import sys
import getopt
import codecs
import tempfile
import shutil
import os

def _find_indentation(line, config):
    if (len(line) and (line[0] in (' ', '\t')) and (not line.isspace())):
        if (line[0] == '\t'):
            config['is-tabs'] = True
        i = 0
        for char in list(line):
            if (char not in (' ', '\t')):
                break
            i += 1
        config['from'] = i

def find_indentation(line, config):
    if (config['from'] < 0):
        _find_indentation(line, config)
    if (config['from'] >= 0):
        indent = (' ' if (not config['is-tabs']) else '\t')
        indent = (indent * config['from'])
        newindent = (' ' if (not config['tabs']) else '\t')
        if (not config['tabs']):
            newindent = (newindent * config['to'])
        return (indent, newindent)
    return False

def replace_inline_tabs(content, config):
    newcontent = ''
    imagined_i = 0
    for i in range(0, len(content)):
        char = content[i]
        if (char == '\t'):
            spaces = (config['tabsize'] - (imagined_i % config['tabsize']))
            newcontent += (' ' * spaces)
            imagined_i += spaces
        else:
            newcontent += char
            imagined_i += 1
    return newcontent

def run(fd_in, fd_out, config):
    while True:
        line = fd_in.readline()
        if (not line):
            break
        line = line.rstrip('\r\n')
        if (config['from'] < 0):
            indent = find_indentation(line, config)
            if (not indent):
                print(line, file=fd_out)
                continue
            (indent, newindent) = indent
        level = 0
        while True:
            whitespace = line[:(len(indent) * (level + 1))]
            if (whitespace == (indent * (level + 1))):
                level += 1
            else:
                break
        content = line[(len(indent) * level):]
        if config['all-tabs']:
            content = replace_inline_tabs(content, config)
        line = ((newindent * level) + content)
        print(line, file=fd_out)

def run_files(filenames, config):
    for filename in filenames:
        with codecs.open(filename, encoding=config['encoding']) as fd_in:
            if config['dry-run']:
                print(('Filename: %s' % filename))
                fd_out = sys.stdout
            else:
                fd_out = tempfile.NamedTemporaryFile(mode='wb', delete=False)
                fd_out.close()
                fd_out = codecs.open(fd_out.name, 'wb', encoding=config['encoding'])
            run(fd_in, fd_out, config)
            if (not config['dry-run']):
                fd_out.close()
                shutil.copy(fd_out.name, filename)
                os.remove(fd_out.name)

def main(args):
    config = {'dry-run': False, 'help': False, 'to': 4, 'from': (- 1), 'tabs': False, 'encoding': 'utf-8', 'is-tabs': False, 'tabsize': 4, 'all-tabs': False}
    possible_args = {'d': 'dry-run', 'h': 'help', 't:': 'to=', 'f:': 'from=', 'n': 'tabs', 'e:': 'encoding=', 's:': 'tabsize=', 'a': 'all-tabs'}
    (optlist, filenames) = getopt.getopt(args[1:], ''.join(possible_args.keys()), possible_args.values())
    (shortargs, longargs) = ([], [])
    for shortarg in possible_args:
        shortargs.append(shortarg.rstrip(':'))
        longargs.append(possible_args[shortarg].rstrip('='))
    for (opt, val) in optlist:
        opt = opt.lstrip('-')
        if (opt in shortargs):
            opt = longargs[shortargs.index(opt)]
        if isinstance(config[opt], bool):
            config[opt] = True
        elif isinstance(config[opt], int):
            config[opt] = int(val)
        else:
            config[opt] = val
    if config['help']:
        help = ("\n        Usage: %s [options] filename(s)\n        Options:\n            -h, --help              Show this message\n            -d, --dry-run           Don't save anything, just print\n                                    the result\n            -t <n>, --to <n>        Convert to this number of spaces\n                                    (default: 4)\n            -f <n>, --from <n>      Convert from this number of spaces\n                                    (default: auto-detect, will also\n                                    detect tabs)\n            -n, --tabs              Don't convert indentation to spaces,\n                                    convert to tabs instead. -t and\n                                    --to will have no effect.\n            -a, --all-tabs          Also convert tabs used for alignment\n                                    in the code (Warning: will replace\n                                    all tabs in the file, even if inside\n                                    a string)\n            -s <n>, --tabsize <n>   Set how many spaces one tab is\n                                    (only has an effect on -a, default: 4)\n            -e <s>, --encoding <s>  Open files with specified encoding\n                                    (default: utf-8)\n        " % args[0])
        print('\n'.join([x[8:] for x in help[1:].split('\n')]))
        sys.exit(0)
    if filenames:
        run_files(filenames, config)
    else:
        run(sys.stdin, sys.stdout, config)
if (__name__ == '__main__'):
    main(sys.argv)