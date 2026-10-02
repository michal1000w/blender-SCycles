/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Translation of compiled OSL camera shaders to the Metal shading language.
 *
 * The input is the bytecode written by the OSL compiler: a list of symbols followed by a list of
 * instructions, with structured control flow encoded as jump targets. Functions are already
 * inlined. The output is one Metal function that executes the same instructions.
 *
 * - Control flow maps to `if` and `for (;;)` statements. An inlined function body is a
 *   `do { } while (false)` block, so that `return` becomes `break`.
 * - Values that depend on the sensor position carry derivatives where the ray differentials
 *   need them, like in the OSL runtime. Two data flow analyses find these symbols.
 * - Parameters with a value assigned by the camera are read from an array at render time.
 * - Strings only exist at translation time: at render time a string is the index of its text,
 *   and instructions that need the text are emitted for each value the string can have.
 * - Image lookups are requests to the kernel, which calls the function again with the result.
 * - Dictionaries are queried by the host during the translation.
 *
 * The one thing that cannot be translated is a string made from a number at render time that
 * the shader then looks at. The translation then fails with a message naming the string. */

#include "device/metal/osl_camera_translate.h"
#include "device/metal/osl_camera_prelude.h"

#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <set>

CCL_NAMESPACE_BEGIN

namespace {

using std::string;
using std::vector;

string format(const char *fmt, ...)
{
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return buffer;
}

/* -------------------------------------------------------------------- */
/* Bytecode */

enum SymKind { SYM_PARAM, SYM_OPARAM, SYM_LOCAL, SYM_TEMP, SYM_GLOBAL, SYM_CONST };
enum Base { B_INT, B_FLOAT, B_VEC, B_MAT, B_STR, B_CLOSURE, B_OTHER };

/* Type of a value in the generated code. */
struct VT {
  Base b;
  /* Carries derivatives. */
  bool d;
};

struct Sym {
  SymKind kind = SYM_LOCAL;
  Base base = B_OTHER;
  /* Element type of a color, as opposed to a point, vector or normal. */
  string type_name;
  /* Zero for scalars. */
  int arraylen = 0;
  string name;
  string cname;

  vector<float> fval;
  vector<int> ival;
  vector<string> sval;

  /* Instructions computing the default value of a parameter. */
  int init_begin = -1;
  int init_end = -1;

  bool used = false;
  int num_writes = 0;
  /* Array that is only ever a copy of this constant array. */
  int alias = -1;
  /* Local array that is only filled with constants: a constant array with these values. */
  bool promoted = false;
  /* Parameter read at render time, with its offset in words. */
  int slot = -1;
  /* Stored in the output structure. */
  bool is_output = false;

  bool has_deriv = false;
  bool needs_deriv = false;
  bool dual = false;

  /* Strings are numbers at render time: indices into the table of all strings of the shader.
   * The analysis finds the values a string symbol can have. A symbol with one value is a
   * constant of the translation. */
  std::set<string> str_values;
  /* The value cannot be known before render time, for example a number formatted as text. */
  bool str_top = false;
  bool str_known = false;
  /* Value assigned by the camera, the default is not computed. */
  bool str_is_overridden = false;
  string str;

  /* First and last instruction reading and writing the symbol, from the compiler. */
  int first_read = INT_MAX;
  int first_write = INT_MAX;
  /* A local string that can be read before it is assigned. */
  bool str_maybe_unassigned = false;

  /* Name of an output in the output structure. */
  string field_name;

  bool str_overridden() const
  {
    return str_is_overridden;
  }

  bool is_array() const
  {
    return arraylen != 0;
  }
  int num_components() const
  {
    return (base == B_VEC) ? 3 : (base == B_MAT) ? 16 : 1;
  }
};

struct Op {
  string name;
  vector<int> args;
  vector<int> jumps;
  string argrw;

  bool writes(const int i) const
  {
    if (i < int(argrw.size())) {
      return argrw[i] == 'w' || argrw[i] == 'W';
    }
    return i == 0;
  }
  bool reads(const int i) const
  {
    if (i < int(argrw.size())) {
      return argrw[i] == 'r' || argrw[i] == 'W';
    }
    return i != 0;
  }
};

struct Token {
  string text;
  bool quoted = false;
  bool hint = false;
};

/* Split a line into whitespace separated tokens. Quoted strings and `%name{...}` hints are
 * single tokens, as both can contain whitespace. */
bool tokenize(const string &line, vector<Token> &tokens)
{
  size_t i = 0;
  const size_t n = line.size();
  while (i < n) {
    const char c = line[i];
    if (c == ' ' || c == '\t' || c == '\r') {
      i++;
      continue;
    }
    Token token;
    if (c == '"') {
      token.quoted = true;
      i++;
      while (i < n && line[i] != '"') {
        if (line[i] == '\\' && i + 1 < n) {
          i++;
          switch (line[i]) {
            case 'n':
              token.text += '\n';
              break;
            case 't':
              token.text += '\t';
              break;
            default:
              token.text += line[i];
              break;
          }
        }
        else {
          token.text += line[i];
        }
        i++;
      }
      if (i >= n) {
        return false;
      }
      i++;
    }
    else if (c == '%') {
      token.hint = true;
      int depth = 0;
      bool in_string = false;
      while (i < n) {
        const char h = line[i];
        if (in_string) {
          if (h == '\\' && i + 1 < n) {
            token.text += h;
            i++;
          }
          else if (h == '"') {
            in_string = false;
          }
        }
        else if (h == '"') {
          in_string = true;
        }
        else if (h == '{') {
          depth++;
        }
        else if (h == '}') {
          depth--;
        }
        else if ((h == ' ' || h == '\t') && depth <= 0) {
          break;
        }
        token.text += line[i];
        i++;
      }
    }
    else {
      while (i < n && line[i] != ' ' && line[i] != '\t' && line[i] != '\r') {
        token.text += line[i];
        i++;
      }
    }
    tokens.push_back(token);
  }
  return true;
}

bool is_integer(const string &text)
{
  if (text.empty()) {
    return false;
  }
  size_t i = (text[0] == '-') ? 1 : 0;
  if (i == text.size()) {
    return false;
  }
  for (; i < text.size(); i++) {
    if (text[i] < '0' || text[i] > '9') {
      return false;
    }
  }
  return true;
}

string float_literal(const float f)
{
  if (std::isnan(f)) {
    return "OSL_NAN";
  }
  if (std::isinf(f)) {
    return (f > 0.0f) ? "OSL_INF" : "(-OSL_INF)";
  }
  string s = format("%.9g", double(f));
  if (s.find_first_of(".e") == string::npos) {
    s += ".0";
  }
  s += "f";
  return (f < 0.0f) ? "(" + s + ")" : s;
}

/* -------------------------------------------------------------------- */
/* Translator */

class Translator {
 public:
  Translator(const OSLCameraTranslateOptions &options) : options(options) {}

  bool translate(const string &bytecode, OSLCameraTranslateResult &result)
  {
    if (!parse(bytecode) || !analyze()) {
      return false;
    }
    return generate(result);
  }

  string error;

 private:
  const OSLCameraTranslateOptions &options;

  vector<Sym> syms;
  std::map<string, int> sym_index;
  vector<Op> ops;
  int main_begin = -1;
  int main_end = -1;

  /* Enclosing loops and inlined functions while generating code. */
  struct Scope {
    bool is_function;
    int id;
    /* Loop instruction, for `continue`. */
    int loop_op;
    /* Functions that returned from inside this loop. */
    std::set<int> returned;
  };
  vector<Scope> scopes;
  int num_scopes = 0;

  bool use_wavelength = false;
  string globals_code;
  vector<OSLCameraTranslateResult::Slot> slots;
  int num_words = 0;

  bool fail(const string &message)
  {
    if (error.empty()) {
      error = message;
    }
    return false;
  }

  /* ------------------------------------------------------------------ */
  /* Parsing */

  static bool parse_type(const string &text, Sym &sym)
  {
    string type = text;
    const size_t bracket = type.find('[');
    if (bracket != string::npos) {
      const string len = type.substr(bracket + 1, type.size() - bracket - 2);
      sym.arraylen = len.empty() ? -1 : atoi(len.c_str());
      type = type.substr(0, bracket);
    }
    sym.type_name = type;
    if (type == "int") {
      sym.base = B_INT;
    }
    else if (type == "float") {
      sym.base = B_FLOAT;
    }
    else if (type == "color" || type == "point" || type == "vector" || type == "normal") {
      sym.base = B_VEC;
    }
    else if (type == "matrix") {
      sym.base = B_MAT;
    }
    else if (type == "string") {
      sym.base = B_STR;
    }
    else if (type == "closure") {
      sym.base = B_CLOSURE;
    }
    else {
      sym.base = B_OTHER;
    }
    return true;
  }

  bool parse(const string &bytecode)
  {
    size_t pos = 0;
    bool header = false;
    bool shader_declared = false;
    int section_param = -1;

    while (pos < bytecode.size()) {
      size_t eol = bytecode.find('\n', pos);
      if (eol == string::npos) {
        eol = bytecode.size();
      }
      const string line = bytecode.substr(pos, eol - pos);
      pos = eol + 1;

      /* Comments quote the source, which can be anything. */
      const size_t line_start = line.find_first_not_of(" \t\r");
      if (line_start == string::npos || line[line_start] == '#') {
        continue;
      }
      vector<Token> tokens;
      if (!tokenize(line, tokens)) {
        return fail("malformed bytecode line: " + line);
      }
      if (tokens.empty()) {
        continue;
      }
      const string &first = tokens[0].text;

      if (!header) {
        if (first != "OpenShadingLanguage") {
          return fail("not OSL bytecode");
        }
        header = true;
        continue;
      }

      /* Instructions are indented, everything else is not. */
      const bool in_code = (line[0] == '\t' || line[0] == ' ');
      if (in_code && !code_started) {
        return fail("instruction outside of a code section");
      }
      if (!in_code && first == "code") {
        if (tokens.size() < 2) {
          return fail("malformed code section");
        }
        close_section(section_param);
        code_started = true;
        section_param = -1;
        if (tokens[1].text == "___main___") {
          main_begin = int(ops.size());
        }
        else {
          const auto it = sym_index.find(tokens[1].text);
          if (it == sym_index.end()) {
            return fail("code section for unknown parameter " + tokens[1].text);
          }
          section_param = it->second;
          syms[section_param].init_begin = int(ops.size());
        }
        continue;
      }

      if (in_code) {
        if (!parse_op(tokens)) {
          return false;
        }
        continue;
      }

      static const char *kinds[] = {"param", "oparam", "local", "temp", "global", "const"};
      int kind = -1;
      for (int i = 0; i < 6; i++) {
        if (first == kinds[i]) {
          kind = i;
        }
      }
      if (kind == -1) {
        if (!shader_declared) {
          /* Shader type and name. */
          shader_declared = true;
          continue;
        }
        return fail("unknown bytecode line: " + line);
      }
      if (!parse_symbol(SymKind(kind), tokens)) {
        return false;
      }
    }

    close_section(section_param);
    if (main_begin == -1) {
      return fail("bytecode has no main code section");
    }
    return true;
  }

  bool code_started = false;

  void close_section(const int section_param)
  {
    if (section_param != -1) {
      syms[section_param].init_end = int(ops.size());
    }
    else if (main_begin != -1 && main_end == -1) {
      main_end = int(ops.size());
    }
  }

  bool parse_symbol(const SymKind kind, const vector<Token> &tokens)
  {
    size_t t = 1;
    if (tokens.size() < 3) {
      return fail("malformed symbol");
    }
    Sym sym;
    sym.kind = kind;
    string type = tokens[t++].text;
    if (type == "closure") {
      /* Followed by the closure type, which carries the array length. Camera shaders have no
       * use for closures: they are computed by no instruction and stored nowhere. */
      const string closure_type = (t < tokens.size()) ? tokens[t++].text : "";
      const size_t bracket = closure_type.find('[');
      type = "closure" + ((bracket != string::npos) ? closure_type.substr(bracket) : "");
    }
    else if (type == "struct") {
      /* Followed by the structure type. Its fields are separate symbols. */
      t++;
      type = "struct";
    }
    parse_type(type, sym);
    if (t >= tokens.size()) {
      return fail("malformed symbol");
    }
    sym.name = tokens[t++].text;

    size_t num_parsed = 0;
    for (; t < tokens.size() && !tokens[t].hint; t++) {
      const Token &token = tokens[t];
      num_parsed++;
      if (sym.base == B_STR) {
        sym.sval.push_back(token.text);
      }
      else if (sym.base == B_INT) {
        sym.ival.push_back(int(strtol(token.text.c_str(), nullptr, 10)));
      }
      else {
        sym.fval.push_back(strtof(token.text.c_str(), nullptr));
      }
    }
    for (; t < tokens.size(); t++) {
      const string &hint = tokens[t].text;
      if (hint.compare(0, 6, "%read{") == 0) {
        sym.first_read = atoi(hint.c_str() + 6);
      }
      else if (hint.compare(0, 7, "%write{") == 0) {
        sym.first_write = atoi(hint.c_str() + 7);
      }
    }
    if (sym.arraylen < 0) {
      /* An array without a length has the length of its default value. */
      sym.arraylen = std::max(int(num_parsed) / sym.num_components(), 1);
    }

    const size_t num_values = size_t(sym.num_components() * std::max(sym.arraylen, 1));
    if (sym.base == B_INT) {
      sym.ival.resize(num_values, 0);
    }
    else if (sym.base == B_STR) {
      sym.sval.resize(num_values);
    }
    else {
      sym.fval.resize(num_values, 0.0f);
    }

    sym_index[sym.name] = int(syms.size());
    syms.push_back(sym);
    return true;
  }

  bool parse_op(const vector<Token> &tokens)
  {
    Op op;
    op.name = tokens[0].text;
    for (size_t t = 1; t < tokens.size(); t++) {
      const Token &token = tokens[t];
      if (token.hint) {
        if (token.text.compare(0, 7, "%argrw{") == 0) {
          for (const char c : token.text.substr(7)) {
            if (c == 'r' || c == 'w' || c == 'W' || c == '-') {
              op.argrw += c;
            }
          }
        }
        continue;
      }
      if (!token.quoted && is_integer(token.text)) {
        op.jumps.push_back(atoi(token.text.c_str()));
        continue;
      }
      const auto it = sym_index.find(token.text);
      if (it == sym_index.end()) {
        return fail("instruction " + op.name + " uses unknown symbol " + token.text);
      }
      op.args.push_back(it->second);
    }
    ops.push_back(op);
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* Analysis */

  /* Instructions in the order they execute: parameter defaults, then the shader. */
  template<typename F> void foreach_live_op(F &&func)
  {
    for (Sym &sym : syms) {
      if (sym.init_begin != -1 && sym.slot == -1 && !sym.str_overridden()) {
        for (int i = sym.init_begin; i < sym.init_end; i++) {
          func(ops[i]);
        }
      }
    }
    for (int i = main_begin; i < main_end; i++) {
      func(ops[i]);
    }
  }

  static bool is_control_op(const string &name)
  {
    return name == "if" || name == "for" || name == "while" || name == "dowhile" ||
           name == "functioncall" || name == "functioncall_nr";
  }

  /* Instructions whose result has no derivatives, whatever the arguments. */
  static bool is_zero_deriv_op(const string &name)
  {
    static const std::set<string> names = {
        "floor",       "ceil",       "round",       "trunc",     "sign",       "step",
        "hashnoise",   "cellnoise",  "splineinverse", "getmessage",
        "texture",     "texture3d",  "environment", "gettextureinfo", "trace",
        "regex_search", "regex_match", "split",     "stoi",      "stof",       "getchar",
        "dict_find",   "dict_next",  "dict_value",  "pointcloud_search", "pointcloud_get",
        "pointcloud_write",
        "eq",          "neq",        "lt",          "le",        "gt",         "ge",
        "getattribute", "raytype",   "arraylength", "Dx",        "Dy",         "Dz",
        "filterwidth", "area",       "calculatenormal", "isnan", "isinf",      "isfinite",
        "bitand",      "bitor",      "xor",         "compl",     "shl",        "shr",
        "and",         "or",         "not",         "logb",
        "wavelength_color", "blackbody", "matrix",  "getmatrix",  "determinant",
        "hash",
        "transpose",   "mxcompref",  "mxcompassign", "strlen",   "startswith", "endswith",
        "isconnected", "isconstant", "backfacing",  "surfacearea", "mod",
    };
    return names.count(name) != 0;
  }

  /* Instructions that use the derivatives of their arguments. */
  static bool is_deriv_query_op(const string &name)
  {
    return name == "Dx" || name == "Dy" || name == "Dz" || name == "filterwidth" ||
           name == "area" || name == "calculatenormal" || name == "texture";
  }

  /* The jump targets of control flow instructions have to describe properly nested ranges of
   * instructions, which everything after this relies on. */
  bool validate_range(const int begin, const int end)
  {
    for (int i = begin; i < end;) {
      const Op &op = ops[i];
      const vector<int> &jumps = op.jumps;
      const size_t expected = (op.name == "if") ? 2 :
                              (op.name == "for" || op.name == "while" || op.name == "dowhile") ?
                                              4 :
                              (op.name == "functioncall" || op.name == "functioncall_nr") ? 1 : 0;
      if (expected == 0) {
        i++;
        continue;
      }
      if (jumps.size() != expected) {
        return fail("malformed instruction " + op.name);
      }
      int previous = i + 1;
      for (const int jump : jumps) {
        if (jump < previous || jump > end) {
          return fail("malformed instruction " + op.name);
        }
        previous = jump;
      }
      /* The ranges inside the instruction. */
      previous = i + 1;
      for (const int jump : jumps) {
        if (!validate_range(previous, jump)) {
          return false;
        }
        previous = jump;
      }
      i = jumps.back();
    }
    return true;
  }

  bool analyze()
  {
    for (const Sym &sym : syms) {
      if (sym.init_begin != -1 && !validate_range(sym.init_begin, sym.init_end)) {
        return false;
      }
    }
    if (!validate_range(main_begin, main_end)) {
      return false;
    }

    /* Parameters with a value assigned by the camera. */
    for (Sym &sym : syms) {
      if (sym.kind != SYM_PARAM || sym.is_array()) {
        continue;
      }
      const auto it = options.params.find(sym.name);
      if (it == options.params.end()) {
        continue;
      }
      const OSLCameraTranslateParam &param = it->second;
      if (param.type == OSLCameraTranslateParam::STRING) {
        if (sym.base == B_STR) {
          sym.sval[0] = param.string_value;
          sym.str_is_overridden = true;
        }
      }
      else if ((param.type == OSLCameraTranslateParam::INT && sym.base == B_INT &&
                param.size == 1) ||
               (param.type == OSLCameraTranslateParam::FLOAT &&
                (sym.base == B_FLOAT || sym.base == B_VEC || sym.base == B_MAT) &&
                param.size == sym.num_components()))
      {
        sym.slot = num_words;
        slots.push_back({sym.name, param.size, num_words, sym.base == B_INT});
        num_words += param.size;
      }
    }

    /* Outputs. */
    for (Sym &sym : syms) {
      if (sym.kind == SYM_OPARAM) {
        sym.is_output = true;
        sym.used = true;
      }
    }

    /* Uses and writes. */
    foreach_live_op([&](const Op &op) {
      for (int i = 0; i < int(op.args.size()); i++) {
        Sym &sym = syms[op.args[i]];
        sym.used = true;
        if (op.writes(i) && !is_control_op(op.name)) {
          sym.num_writes++;
        }
      }
    });

    /* Local arrays that are only a copy of a constant array read the constant directly. Lens
     * shaders hold their tables in such arrays, copying them for every ray would be slow. */
    foreach_live_op([&](const Op &op) {
      if ((op.name == "assign" || op.name == "arraycopy") && op.args.size() == 2) {
        Sym &dst = syms[op.args[0]];
        const Sym &src = syms[op.args[1]];
        if (dst.is_array() && dst.num_writes == 1 && src.kind == SYM_CONST && src.is_array() &&
            (dst.kind == SYM_LOCAL || dst.kind == SYM_TEMP) && dst.base == src.base &&
            dst.arraylen == src.arraylen)
        {
          dst.alias = op.args[1];
        }
      }
    });

    promote_constant_arrays();

    analyze_strings();
    analyze_messages();

    analyze_derivatives();
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* Strings */

  /* Strings are numbers at render time: the index of the string in a table that only exists
   * during the translation. Every instruction that looks into a string is evaluated by the
   * translator. For that the analysis finds the set of values each string symbol can have:
   * - One value: the symbol is a constant.
   * - A few values: instructions using the string are emitted once for each value, selected at
   *   render time by the index.
   * - Unknown (`str_top`), for a number formatted as text or a string growing in a loop: the
   *   string can only be printed, which does nothing on this device. */

  static constexpr size_t STR_VALUES_MAX = 256;
  /* Largest number of versions of one instruction for the values of its strings. */
  static constexpr size_t STR_VERSIONS_MAX = 512;

  vector<string> str_table;
  std::map<string, int> str_ids;
  /* Values of string symbols while emitting one version of an instruction. */
  std::map<int, string> str_fixed;

  int str_id(const string &value)
  {
    const auto it = str_ids.find(value);
    if (it != str_ids.end()) {
      return it->second;
    }
    const int id = int(str_table.size());
    str_table.push_back(value);
    str_ids[value] = id;
    return id;
  }

  string str_literal(const string &value)
  {
    return format("%d", str_id(value));
  }

  static bool str_add(Sym &sym, const string &value)
  {
    if (sym.str_top || sym.str_values.count(value)) {
      return false;
    }
    if (sym.str_values.size() >= STR_VALUES_MAX) {
      sym.str_top = true;
      sym.str_values.clear();
      return true;
    }
    sym.str_values.insert(value);
    return true;
  }

  static bool str_make_top(Sym &sym)
  {
    if (sym.str_top) {
      return false;
    }
    sym.str_top = true;
    sym.str_values.clear();
    return true;
  }

  static bool str_union(Sym &dst, const Sym &src)
  {
    if (src.str_top) {
      return str_make_top(dst);
    }
    bool changed = false;
    for (const string &value : src.str_values) {
      changed |= str_add(dst, value);
    }
    return changed;
  }

  /* Find the local strings that can be read before they are assigned: those are empty then.
   * `assigned` are the symbols that are assigned on every path to the current instruction. */
  void scan_assigned(const int begin,
                     const int end,
                     std::set<int> &assigned,
                     std::set<int> *return_assigned,
                     bool &has_return)
  {
    const auto intersect = [](std::set<int> &a, const std::set<int> &b) {
      for (auto it = a.begin(); it != a.end();) {
        it = b.count(*it) ? std::next(it) : a.erase(it);
      }
    };
    for (int i = begin; i < end;) {
      const Op &op = ops[i];
      if (op.name == "if" && op.jumps.size() == 2) {
        std::set<int> then_assigned = assigned;
        std::set<int> else_assigned = assigned;
        scan_assigned(i + 1, op.jumps[0], then_assigned, return_assigned, has_return);
        scan_assigned(op.jumps[0], op.jumps[1], else_assigned, return_assigned, has_return);
        intersect(then_assigned, else_assigned);
        assigned = then_assigned;
        i = std::max(op.jumps[1], i + 1);
      }
      else if ((op.name == "for" || op.name == "while" || op.name == "dowhile") &&
               op.jumps.size() == 4)
      {
        scan_assigned(i + 1, op.jumps[0], assigned, return_assigned, has_return);
        /* What a loop assigns is not relied on after it. */
        std::set<int> inside = assigned;
        if (op.name == "dowhile") {
          scan_assigned(op.jumps[1], op.jumps[3], inside, return_assigned, has_return);
          scan_assigned(op.jumps[0], op.jumps[1], inside, return_assigned, has_return);
        }
        else {
          scan_assigned(op.jumps[0], op.jumps[3], inside, return_assigned, has_return);
        }
        i = std::max(op.jumps[3], i + 1);
      }
      else if ((op.name == "functioncall" || op.name == "functioncall_nr") &&
               op.jumps.size() == 1)
      {
        /* A return skips the rest of its function. */
        std::set<int> at_return;
        bool returns = false;
        scan_assigned(i + 1, op.jumps[0], assigned, &at_return, returns);
        if (returns) {
          intersect(assigned, at_return);
        }
        i = std::max(op.jumps[0], i + 1);
      }
      else {
        if (op.name == "return" && return_assigned) {
          if (has_return) {
            intersect(*return_assigned, assigned);
          }
          else {
            *return_assigned = assigned;
            has_return = true;
          }
        }
        for (int arg = 0; arg < int(op.args.size()); arg++) {
          Sym &sym = syms[op.args[arg]];
          if (sym.base == B_STR && sym.kind == SYM_LOCAL && !sym.is_array() && op.reads(arg) &&
              !assigned.count(op.args[arg]))
          {
            sym.str_maybe_unassigned = true;
          }
        }
        for (int arg = 0; arg < int(op.args.size()); arg++) {
          const Sym &sym = syms[op.args[arg]];
          if (sym.base == B_STR && sym.kind == SYM_LOCAL && !sym.is_array() && op.writes(arg)) {
            assigned.insert(op.args[arg]);
          }
        }
        i++;
      }
    }
  }

  /* The values of the string arguments of an instruction from `first` on, for each combination
   * of the values the arguments can have. False if a value is unknown or there are too many. */
  struct StrCombos {
    /* One entry per combination: the value of each argument, in order. */
    vector<vector<string>> values;
  };

  bool str_combinations(const Op &op, const vector<int> &args, StrCombos &combos) const
  {
    combos.values.assign(1, {});
    for (const int arg : args) {
      const Sym &sym = syms[op.args[arg]];
      if (sym.str_top || sym.is_array()) {
        return false;
      }
      vector<vector<string>> next;
      for (const vector<string> &prefix : combos.values) {
        for (const string &value : sym.str_values) {
          next.push_back(prefix);
          next.back().push_back(value);
          if (next.size() > STR_VERSIONS_MAX) {
            return false;
          }
        }
      }
      combos.values.swap(next);
    }
    return true;
  }

  /* Number in the default formatting of the OSL runtime. */
  static string c_format(const string &spec, const double value)
  {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), spec.c_str(), value);
    return buffer;
  }
  static string c_format(const string &spec, const int value)
  {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), spec.c_str(), value);
    return buffer;
  }

  /* The result of format(): `values` are the values of the string arguments from `first` on,
   * in order, the first being the format itself. Numbers have to be constants. */
  bool format_string(const Op &op, const int first, const vector<string> &values, string &result)
      const
  {
    result.clear();
    const string &fmt = values[0];
    size_t next_string = 1;
    int arg = first + 1;
    for (size_t i = 0; i < fmt.size();) {
      if (fmt[i] != '%') {
        result += fmt[i++];
        continue;
      }
      if (i + 1 < fmt.size() && fmt[i + 1] == '%') {
        result += '%';
        i += 2;
        continue;
      }
      const size_t begin = i;
      while (i < fmt.size() && !strchr("cdefgimnopsuvxX", fmt[i])) {
        i++;
      }
      if (i >= fmt.size() || arg >= int(op.args.size())) {
        return false;
      }
      i++;
      string spec = fmt.substr(begin, i - begin);
      char &conversion = spec.back();
      /* Length modifiers have no meaning for the types of the shading language. */
      for (size_t c = 1; c + 1 < spec.size();) {
        if (strchr("hlLqjzt*", spec[c])) {
          spec.erase(c, 1);
        }
        else {
          c++;
        }
      }
      const Sym &sym = syms[op.args[arg++]];
      if (sym.base == B_STR) {
        if (sym.is_array() || next_string >= values.size()) {
          return false;
        }
        conversion = 's';
        char buffer[4096];
        snprintf(buffer, sizeof(buffer), spec.c_str(), values[next_string++].c_str());
        result += buffer;
        continue;
      }
      if (sym.kind != SYM_CONST) {
        return false;
      }
      const int count = sym.num_components() * std::max(sym.arraylen, 1);
      if (sym.base == B_INT) {
        if (!strchr("dioxXu", conversion)) {
          conversion = 'd';
        }
        for (int c = 0; c < count; c++) {
          result += ((c > 0) ? " " : "") + c_format(spec, sym.ival[c]);
        }
      }
      else if (sym.base == B_FLOAT || sym.base == B_VEC || sym.base == B_MAT) {
        if (!strchr("fgeEG", conversion)) {
          conversion = 'f';
        }
        for (int c = 0; c < count; c++) {
          result += ((c > 0) ? " " : "") + c_format(spec, double(sym.fval[c]));
        }
      }
      else {
        return false;
      }
    }
    return true;
  }

  /* substr() of the OSL runtime. */
  static string str_substr(const string &value, const int start, const int length)
  {
    const int slen = int(value.size());
    if (slen == 0) {
      return "";
    }
    int b = start;
    if (b < 0) {
      b += slen;
    }
    b = std::min(std::max(b, 0), slen);
    return value.substr(size_t(b), size_t(std::min(std::max(length, 0), slen)));
  }

  /* split() of the OSL runtime: the pieces that are stored in the result array. At most
   * `maxsplit` pieces are made, the last one is the rest of the string. */
  static vector<string> str_split(const string &str, const string &sep, int maxsplit, const int len)
  {
    maxsplit = std::min(std::max(maxsplit, 0), len);
    vector<string> pieces;
    if (maxsplit == 0) {
      return pieces;
    }
    const size_t size = str.size();
    if (sep.empty()) {
      size_t i = 0, j = 0;
      while (i < size) {
        while (i < size && isspace((unsigned char)str[i])) {
          i++;
        }
        j = i;
        while (i < size && !isspace((unsigned char)str[i])) {
          i++;
        }
        if (j < i) {
          if (int(pieces.size()) + 1 >= maxsplit) {
            break;
          }
          pieces.push_back(str.substr(j, i - j));
          while (i < size && isspace((unsigned char)str[i])) {
            i++;
          }
          j = i;
        }
      }
      if (j < size) {
        pieces.push_back(str.substr(j));
      }
    }
    else {
      const size_t n = sep.size();
      size_t i = 0, j = 0;
      while (i + n <= size) {
        if (str[i] == sep[0] && str.compare(i, n, sep) == 0) {
          if (int(pieces.size()) + 1 >= maxsplit) {
            break;
          }
          pieces.push_back(str.substr(j, i - j));
          i = j = i + n;
        }
        else {
          i++;
        }
      }
      pieces.push_back(str.substr(j));
    }
    return pieces;
  }

  /* Instructions that compute a string from other strings. */
  static bool is_string_function_op(const string &name)
  {
    return name == "concat" || name == "format" || name == "substr";
  }

  /* The values that the result of a string function can have. False if unknown. */
  bool string_function_values(const Op &op, std::set<string> &result) const
  {
    vector<int> args;
    for (int i = 1; i < int(op.args.size()); i++) {
      if (syms[op.args[i]].base == B_STR) {
        args.push_back(i);
      }
    }
    StrCombos combos;
    if (!str_combinations(op, args, combos)) {
      return false;
    }
    for (const vector<string> &values : combos.values) {
      if (op.name == "concat") {
        string value;
        for (const string &part : values) {
          value += part;
        }
        result.insert(value);
      }
      else if (op.name == "format") {
        string value;
        if (values.empty() || syms[op.args[1]].base != B_STR ||
            !format_string(op, 1, values, value))
        {
          return false;
        }
        result.insert(value);
      }
      else if (op.name == "substr") {
        if (op.args.size() != 4 || values.size() != 1) {
          return false;
        }
        const Sym &start = syms[op.args[2]];
        const Sym &length = syms[op.args[3]];
        if (start.base != B_INT || length.base != B_INT) {
          return false;
        }
        if (start.kind == SYM_CONST && length.kind == SYM_CONST) {
          result.insert(str_substr(values[0], start.ival[0], length.ival[0]));
        }
        else {
          /* Any part of the string. */
          const string &value = values[0];
          result.insert("");
          for (size_t b = 0; b < value.size(); b++) {
            for (size_t n = 1; b + n <= value.size(); n++) {
              result.insert(value.substr(b, n));
            }
          }
        }
      }
      if (result.size() > STR_VALUES_MAX) {
        return false;
      }
    }
    return true;
  }

  /* Dictionaries. The host evaluates the queries while translating: the shader then only
   * handles node numbers, and reads the results from tables. */
  static constexpr int DICT_NODES_MAX = 4096;

  /* Evaluate the queries of the shader for the values that their strings can have so far.
   * Returns true if that found new nodes. */
  bool analyze_dictionaries()
  {
    OSLCameraDictionary *dictionary = options.dictionary;
    if (dictionary == nullptr) {
      return false;
    }
    const int num_before = dictionary->num_nodes();
    foreach_live_op([&](const Op &op) {
      if (op.name != "dict_find" || op.args.size() != 3) {
        return;
      }
      const bool by_name = (syms[op.args[1]].base == B_STR);
      StrCombos combos;
      if (!str_combinations(op, by_name ? vector<int>{1, 2} : vector<int>{2}, combos)) {
        return;
      }
      for (const vector<string> &values : combos.values) {
        if (by_name) {
          dictionary->find(values[0], values[1]);
        }
        else {
          /* Relative to any node, including the ones this finds. */
          for (int node = 1; node <= dictionary->num_nodes() && node <= DICT_NODES_MAX; node++) {
            dictionary->find(node, values[0]);
          }
        }
      }
    });
    return dictionary->num_nodes() != num_before;
  }

  /* The strings that dict_value() can return. */
  bool dict_string_values(const Op &op, Sym &dst)
  {
    OSLCameraDictionary *dictionary = options.dictionary;
    StrCombos combos;
    if (dictionary == nullptr || op.args.size() != 4 || dst.is_array() ||
        !str_combinations(op, {2}, combos))
    {
      return str_make_top(dst);
    }
    bool changed = false;
    for (const vector<string> &values : combos.values) {
      for (int node = 1; node <= dictionary->num_nodes(); node++) {
        string text;
        if (dictionary->value(node, values[0], text)) {
          changed |= str_add(dst, text);
        }
      }
    }
    return changed;
  }

  int num_dict_tables = 0;

  bool emit_dict(const Op &op, string &code)
  {
    OSLCameraDictionary *dictionary = options.dictionary;
    const int nargs = int(op.args.size());
    if (dictionary == nullptr) {
      return fail("instruction " + op.name + " is not supported on this device");
    }
    const int num_nodes = dictionary->num_nodes();
    if (num_nodes > DICT_NODES_MAX) {
      return fail("instruction " + op.name + ": the dictionary has too many nodes");
    }
    /* Node number that is valid for the tables: 0 is no node. */
    const auto node_index = [&](const int arg, string &result) {
      string node;
      if (!read(op.args[arg], {B_INT, false}, node)) {
        return false;
      }
      result = "((uint(" + node + format(") <= %du) ? ", num_nodes) + node + " : 0)";
      return true;
    };
    const auto int_table = [&](const vector<int> &values) {
      const string name = format("osl_dict_%d", num_dict_tables++);
      globals_code += "constant int " + name + format("[%d] = {", int(values.size()));
      for (size_t i = 0; i < values.size(); i++) {
        globals_code += format("%d", values[i]) + ((i + 1 < values.size()) ? ", " : "");
      }
      globals_code += "};\n";
      return name;
    };

    if (op.name == "dict_find") {
      /* dict_find result dictionary|node query */
      string query;
      if (nargs != 3 || !string_arg(op, 2, query)) {
        return false;
      }
      if (syms[op.args[1]].base == B_STR) {
        string name;
        if (!string_arg(op, 1, name)) {
          return false;
        }
        return write(op.args[0], format("%d", dictionary->find(name, query)), {B_INT, false}, code);
      }
      vector<int> found(num_nodes + 1, 0);
      for (int node = 1; node <= num_nodes; node++) {
        found[node] = dictionary->find(node, query);
      }
      if (dictionary->num_nodes() != num_nodes) {
        return fail("instruction dict_find: the dictionary changed during translation");
      }
      string index;
      if (!node_index(1, index)) {
        return false;
      }
      return write(op.args[0], int_table(found) + "[" + index + "]", {B_INT, false}, code);
    }
    if (op.name == "dict_next") {
      string index;
      if (nargs != 2 || !node_index(1, index)) {
        return fail("malformed instruction dict_next");
      }
      vector<int> next(num_nodes + 1, 0);
      for (int node = 1; node <= num_nodes; node++) {
        next[node] = dictionary->next(node);
      }
      return write(op.args[0], int_table(next) + "[" + index + "]", {B_INT, false}, code);
    }

    /* dict_value result node attribute value */
    string attribute, index;
    if (nargs != 4 || !string_arg(op, 2, attribute) || !node_index(1, index)) {
      return false;
    }
    const Sym &value = syms[op.args[3]];
    const int components = value.num_components() * std::max(value.arraylen, 1);
    const bool is_string = (value.base == B_STR && !value.is_array());
    const bool is_int = (value.base == B_INT);
    const bool is_float = (value.base == B_FLOAT || value.base == B_VEC || value.base == B_MAT);
    if ((!is_string && !is_int && !is_float) || value.kind == SYM_CONST || value.alias != -1 ||
        value.promoted)
    {
      return write(op.args[0], "0", {B_INT, false}, code);
    }

    vector<int> found(num_nodes + 1, 0);
    vector<int> ints((num_nodes + 1) * components, 0);
    vector<float> floats((num_nodes + 1) * components, 0.0f);
    for (int node = 1; node <= num_nodes; node++) {
      string text;
      if (!dictionary->value(node, attribute, text)) {
        continue;
      }
      found[node] = 1;
      if (is_string) {
        ints[node] = str_id(text);
        continue;
      }
      /* Numbers separated by white space or commas. */
      const char *p = text.c_str();
      for (int c = 0; c < components; c++) {
        char *end = nullptr;
        if (is_int) {
          const long v = strtol(p, &end, 10);
          if (end != p) {
            ints[node * components + c] = int(v);
          }
        }
        else {
          const float v = strtof(p, &end);
          if (end != p) {
            floats[node * components + c] = v;
          }
        }
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
          p++;
        }
        if (*p == ',') {
          p++;
        }
      }
    }

    string table;
    if (is_float) {
      table = format("osl_dict_%d", num_dict_tables++);
      globals_code += "constant float " + table + format("[%d] = {", int(floats.size()));
      for (size_t i = 0; i < floats.size(); i++) {
        globals_code += float_literal(floats[i]) + ((i + 1 < floats.size()) ? ", " : "");
      }
      globals_code += "};\n";
    }
    else {
      table = int_table(ints);
    }
    const string found_table = int_table(found);

    const auto component = [&](const int c) {
      return table + format("[dict_node * %d + %d]", components, c);
    };
    /* One element of the value, from the components starting at `c`. */
    const auto element_value = [&](const int c) {
      if (value.base == B_VEC) {
        return "float3(" + component(c) + ", " + component(c + 1) + ", " + component(c + 2) + ")";
      }
      if (value.base == B_MAT) {
        string m = "osl_mat(";
        for (int i = 0; i < 16; i++) {
          m += component(c + i) + ((i < 15) ? ", " : ")");
        }
        return m;
      }
      return component(c);
    };
    string assign;
    if (value.is_array()) {
      for (int i = 0; i < value.arraylen; i++) {
        string converted;
        if (!convert(element_value(i * value.num_components()), {value.base, false}, vt(value),
                     converted))
        {
          return false;
        }
        assign += value.cname + format("[%d] = ", i) + converted + ";\n";
      }
    }
    else if (!(is_string && (value.str_known || value.str_top)) &&
             !write(op.args[3], element_value(0), {value.base, false}, assign))
    {
      return false;
    }
    string found_result, missing_result;
    if (!write(op.args[0], "1", {B_INT, false}, found_result) ||
        !write(op.args[0], "0", {B_INT, false}, missing_result))
    {
      return false;
    }
    code += "{\nconst int dict_node = " + index + ";\nif (" + found_table +
            "[dict_node] != 0) {\n" + assign + found_result + "}\nelse {\n" + missing_result +
            "}\n}\n";
    return true;
  }

  void analyze_strings()
  {
    for (const Sym &sym : syms) {
      if (sym.init_begin != -1 && sym.slot == -1 && !sym.str_overridden()) {
        std::set<int> assigned;
        bool has_return = false;
        scan_assigned(sym.init_begin, sym.init_end, assigned, nullptr, has_return);
      }
    }
    {
      std::set<int> assigned;
      bool has_return = false;
      scan_assigned(main_begin, main_end, assigned, nullptr, has_return);
    }

    /* The empty string is the first of the table: a string that was never assigned. */
    str_id("");

    for (Sym &sym : syms) {
      if (sym.base != B_STR) {
        continue;
      }
      if (sym.kind == SYM_CONST) {
        for (const string &value : sym.sval) {
          str_add(sym, value);
        }
      }
      else if (sym.kind == SYM_PARAM || sym.kind == SYM_OPARAM) {
        /* The default is replaced by the result of the instructions computing it. */
        if (sym.init_begin == -1 || sym.str_is_overridden || sym.is_array()) {
          for (const string &value : sym.sval) {
            str_add(sym, value);
          }
        }
      }
      else if (sym.kind == SYM_GLOBAL) {
        str_add(sym, "");
      }
      else if (sym.kind == SYM_LOCAL) {
        /* Empty until it is assigned. */
        if (sym.is_array() || sym.str_maybe_unassigned) {
          str_add(sym, "");
        }
      }
      else if (sym.is_array()) {
        str_add(sym, "");
      }
    }

    bool changed = true;
    while (changed) {
      changed = analyze_dictionaries();
      foreach_live_op([&](const Op &op) {
        if (is_control_op(op.name)) {
          return;
        }
        const string &name = op.name;
        const int nargs = int(op.args.size());
        for (int i = 0; i < nargs; i++) {
          Sym &dst = syms[op.args[i]];
          if (dst.base != B_STR || !op.writes(i) || dst.kind == SYM_CONST) {
            continue;
          }
          if ((name == "assign" || name == "arraycopy") && i == 0 && nargs == 2 &&
              syms[op.args[1]].base == B_STR)
          {
            changed |= str_union(dst, syms[op.args[1]]);
          }
          else if (name == "aref" && i == 0 && nargs == 3 && syms[op.args[1]].base == B_STR) {
            changed |= str_union(dst, syms[op.args[1]]);
          }
          else if (name == "aassign" && i == 0 && nargs == 3 && syms[op.args[2]].base == B_STR) {
            changed |= str_union(dst, syms[op.args[2]]);
          }
          else if (is_string_function_op(name) && i == 0) {
            std::set<string> values;
            if (string_function_values(op, values)) {
              for (const string &value : values) {
                changed |= str_add(dst, value);
              }
            }
            else {
              changed |= str_make_top(dst);
            }
          }
          else if (name == "split" && i == 2 && nargs >= 3) {
            /* split count string results [separator [maxsplit]] */
            vector<int> args = {1};
            if (nargs > 3) {
              args.push_back(3);
            }
            StrCombos combos;
            if (!str_combinations(op, args, combos)) {
              changed |= str_make_top(dst);
              continue;
            }
            for (const vector<string> &values : combos.values) {
              const string sep = (values.size() > 1) ? values[1] : "";
              /* Fewer splits only give other last pieces. */
              for (int maxsplit = 0; maxsplit <= dst.arraylen; maxsplit++) {
                for (const string &piece : str_split(values[0], sep, maxsplit, dst.arraylen)) {
                  changed |= str_add(dst, piece);
                }
              }
            }
          }
          else if (name == "getmessage") {
            /* Any string that was sent as a message. */
            foreach_live_op([&](const Op &other) {
              if (other.name == "setmessage" && other.args.size() == 2 &&
                  syms[other.args[1]].base == B_STR)
              {
                changed |= str_union(dst, syms[other.args[1]]);
              }
            });
          }
          else if (name == "getattribute" || name == "texture" || name == "texture3d" ||
                   name == "environment" || name == "pointcloud_get" ||
                   name == "pointcloud_search" || name == "dict_value")
          {
            /* No attribute of the camera is a string, error messages are empty, and the
             * dictionary values are found by `dict_string_values()`. */
            if (name == "dict_value") {
              changed |= dict_string_values(op, dst);
            }
            else if (name != "getattribute") {
              changed |= str_add(dst, "");
            }
          }
          else {
            changed |= str_make_top(dst);
          }
        }
      });
    }

    for (Sym &sym : syms) {
      if (sym.base != B_STR) {
        continue;
      }
      if (!sym.str_top && sym.str_values.empty()) {
        /* Never assigned a value. */
        sym.str_values.insert("");
      }
      for (const string &value : sym.str_values) {
        str_id(value);
      }
      sym.str_known = !sym.is_array() && !sym.str_top && sym.str_values.size() == 1 &&
                      !sym.is_output;
      if (sym.str_known) {
        sym.str = *sym.str_values.begin();
      }
    }
  }

  /* Value of a string argument during the translation. */
  bool string_value(const int sym_index, string &value) const
  {
    const auto it = str_fixed.find(sym_index);
    if (it != str_fixed.end()) {
      value = it->second;
      return true;
    }
    const Sym &sym = syms[sym_index];
    if (sym.base == B_STR && sym.str_known) {
      value = sym.str;
      return true;
    }
    return false;
  }

  /* Local arrays that are filled element by element with constants, each element once, become
   * constant arrays. Shaders hold tables and images in such arrays: an array of colors with
   * initial values is compiled to one assignment per element, which must not run for every ray.
   * Elements read before the shader assigns them have their final value as well, which only
   * differs for shaders that read uninitialized elements. */
  void promote_constant_arrays()
  {
    std::map<int, vector<char>> assigned;
    std::set<int> rejected;

    foreach_live_op([&](const Op &op) {
      if (is_control_op(op.name)) {
        return;
      }
      for (int i = 0; i < int(op.args.size()); i++) {
        const int index = op.args[i];
        Sym &array = syms[index];
        if (!op.writes(i) || !array.is_array() || rejected.count(index)) {
          continue;
        }
        bool constant = false;
        if (op.name == "aassign" && i == 0 && op.args.size() == 3 && array.arraylen > 0 &&
            array.alias == -1 && (array.kind == SYM_LOCAL || array.kind == SYM_TEMP) &&
            (array.base == B_INT || array.base == B_FLOAT || array.base == B_VEC))
        {
          const Sym &element = syms[op.args[1]];
          const Sym &value = syms[op.args[2]];
          const bool convertible = (value.base == array.base) ||
                                   (value.base == B_INT && array.base != B_INT) ||
                                   (value.base == B_FLOAT && array.base == B_VEC);
          if (element.kind == SYM_CONST && element.base == B_INT && !element.is_array() &&
              value.kind == SYM_CONST && !value.is_array() && convertible)
          {
            const int at = element.ival[0];
            vector<char> &flags = assigned[index];
            flags.resize(array.arraylen, 0);
            if (at >= 0 && at < array.arraylen && !flags[at]) {
              flags[at] = 1;
              constant = true;
            }
          }
        }
        if (!constant) {
          rejected.insert(index);
        }
      }
    });

    for (const auto &it : assigned) {
      if (rejected.count(it.first)) {
        continue;
      }
      Sym &array = syms[it.first];
      array.promoted = true;
      const int components = array.num_components();
      array.fval.assign((array.base == B_INT) ? 0 : size_t(components * array.arraylen), 0.0f);
      array.ival.assign((array.base == B_INT) ? size_t(array.arraylen) : 0, 0);
    }
    if (assigned.empty()) {
      return;
    }

    /* The values. */
    foreach_live_op([&](const Op &op) {
      if (op.name != "aassign" || op.args.size() != 3 || !syms[op.args[0]].promoted) {
        return;
      }
      Sym &array = syms[op.args[0]];
      const int at = syms[op.args[1]].ival[0];
      const Sym &value = syms[op.args[2]];
      if (array.base == B_INT) {
        array.ival[at] = value.ival[0];
      }
      else if (array.base == B_FLOAT) {
        array.fval[at] = (value.base == B_INT) ? float(value.ival[0]) : value.fval[0];
      }
      else {
        for (int c = 0; c < 3; c++) {
          array.fval[at * 3 + c] = (value.base == B_INT)   ? float(value.ival[0]) :
                                   (value.base == B_FLOAT) ? value.fval[0] :
                                                             value.fval[c];
        }
      }
    });
  }

  bool explicit_derivs = false;

  /* The outputs that make the ray. */
  bool is_renderer_output(const string &name) const
  {
    if (name == "position" || name == "direction" || name == "throughput") {
      return true;
    }
    return explicit_derivs &&
           (name == "dPdx" || name == "dPdy" || name == "dDdx" || name == "dDdy");
  }

  /* Noise that can be Gabor noise. */
  bool is_gabor_op(const Op &op) const
  {
    if ((op.name != "noise" && op.name != "pnoise") || op.args.size() < 2) {
      return false;
    }
    const Sym &type = syms[op.args[1]];
    return type.base == B_STR && !type.is_array() && type.str_values.count("gabor") != 0;
  }

  static bool carries_derivs(const Sym &sym)
  {
    return sym.base == B_FLOAT || sym.base == B_VEC;
  }

  void analyze_derivatives()
  {
    explicit_derivs = false;
    for (const char *name : {"dPdx", "dPdy", "dDdx", "dDdy"}) {
      const auto it = sym_index.find(name);
      if (it != sym_index.end() && syms[it->second].kind == SYM_OPARAM) {
        explicit_derivs = true;
      }
    }

    for (Sym &sym : syms) {
      /* The sensor position is the only input with derivatives. */
      if (sym.kind == SYM_GLOBAL && sym.name == "P") {
        sym.has_deriv = true;
      }
      if (!explicit_derivs && sym.kind == SYM_OPARAM &&
          (sym.name == "position" || sym.name == "direction"))
      {
        sym.needs_deriv = true;
      }
    }

    bool changed = true;
    while (changed) {
      changed = false;
      foreach_live_op([&](const Op &op) {
        if (is_control_op(op.name)) {
          return;
        }
        if (is_deriv_query_op(op.name)) {
          for (int i = 0; i < int(op.args.size()); i++) {
            Sym &sym = syms[op.args[i]];
            if (op.reads(i) && carries_derivs(sym) && !sym.needs_deriv) {
              sym.needs_deriv = true;
              changed = true;
            }
          }
          return;
        }
        if (is_zero_deriv_op(op.name)) {
          return;
        }
        if (is_gabor_op(op)) {
          /* Gabor noise filters with the derivatives of the position. */
          for (int i = 1; i < int(op.args.size()); i++) {
            Sym &sym = syms[op.args[i]];
            if (carries_derivs(sym) && !sym.needs_deriv) {
              sym.needs_deriv = true;
              changed = true;
            }
          }
        }
        bool read_has = false;
        bool write_needs = false;
        for (int i = 0; i < int(op.args.size()); i++) {
          const Sym &sym = syms[op.args[i]];
          if (!carries_derivs(sym)) {
            continue;
          }
          read_has |= op.reads(i) && sym.has_deriv;
          write_needs |= op.writes(i) && sym.needs_deriv;
        }
        for (int i = 0; i < int(op.args.size()); i++) {
          Sym &sym = syms[op.args[i]];
          if (!carries_derivs(sym)) {
            continue;
          }
          if (op.writes(i) && read_has && !sym.has_deriv) {
            sym.has_deriv = true;
            changed = true;
          }
          if (op.reads(i) && write_needs && !sym.needs_deriv) {
            sym.needs_deriv = true;
            changed = true;
          }
        }
      });
    }

    for (Sym &sym : syms) {
      sym.dual = carries_derivs(sym) && sym.has_deriv && sym.needs_deriv &&
                 sym.kind != SYM_CONST && sym.slot == -1 && sym.alias == -1 && !sym.promoted;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Types and conversions */

  static string ctype(const VT t)
  {
    switch (t.b) {
      case B_INT:
        return "int";
      case B_FLOAT:
        return t.d ? "DualF" : "float";
      case B_VEC:
        return t.d ? "DualV" : "float3";
      case B_MAT:
        return "OslMat";
      default:
        return "int";
    }
  }

  static string zero(const VT t)
  {
    switch (t.b) {
      case B_INT:
        return "0";
      case B_FLOAT:
        return t.d ? "osl_dual(0.0f)" : "0.0f";
      case B_VEC:
        return t.d ? "osl_dual(float3(0.0f))" : "float3(0.0f)";
      case B_MAT:
        return "osl_mat_diag(0.0f)";
      default:
        return "0";
    }
  }

  /* Convert an expression of type `from` to type `to`. */
  bool convert(const string &expr, const VT from, const VT to, string &result)
  {
    if (from.b == to.b) {
      if (from.d == to.d || from.b == B_INT || from.b == B_MAT) {
        result = expr;
      }
      else if (to.d) {
        result = "osl_dual(" + expr + ")";
      }
      else {
        result = "(" + expr + ").v";
      }
      return true;
    }
    if (from.b == B_INT && to.b == B_FLOAT) {
      result = "float(" + expr + ")";
      if (to.d) {
        result = "osl_dual(" + result + ")";
      }
      return true;
    }
    if (from.b == B_INT && to.b == B_VEC) {
      result = "float3(float(" + expr + "))";
      if (to.d) {
        result = "osl_dual(" + result + ")";
      }
      return true;
    }
    if (from.b == B_FLOAT && to.b == B_VEC) {
      if (from.d && to.d) {
        result = "osl_dual3(" + expr + ")";
      }
      else if (from.d) {
        result = "float3((" + expr + ").v)";
      }
      else if (to.d) {
        result = "osl_dual(float3(" + expr + "))";
      }
      else {
        result = "float3(" + expr + ")";
      }
      return true;
    }
    if (from.b == B_FLOAT && to.b == B_INT) {
      result = from.d ? "osl_ftoi((" + expr + ").v)" : "osl_ftoi(" + expr + ")";
      return true;
    }
    if ((from.b == B_INT || from.b == B_FLOAT) && to.b == B_MAT) {
      string f;
      convert(expr, from, {B_FLOAT, false}, f);
      result = "osl_mat_diag(" + f + ")";
      return true;
    }
    return fail("unsupported type conversion in shader");
  }

  static VT vt(const Sym &sym)
  {
    return {sym.base, sym.dual};
  }

  string literal(const Sym &sym, const int element)
  {
    switch (sym.base) {
      case B_INT:
        /* The most negative number is not a literal. */
        return (sym.ival[element] == INT_MIN) ? "(-2147483647 - 1)" :
                                                format("%d", sym.ival[element]);
      case B_STR:
        return str_literal(sym.sval[element]);
      case B_FLOAT:
        return float_literal(sym.fval[element]);
      case B_VEC: {
        const float *f = &sym.fval[element * 3];
        return "float3(" + float_literal(f[0]) + ", " + float_literal(f[1]) + ", " +
               float_literal(f[2]) + ")";
      }
      case B_MAT: {
        const float *f = &sym.fval[element * 16];
        string s = "osl_mat(";
        for (int i = 0; i < 16; i++) {
          s += float_literal(f[i]) + ((i < 15) ? ", " : ")");
        }
        return s;
      }
      default:
        return "0";
    }
  }

  /* Expression for the value of a scalar symbol. */
  string expr(const Sym &sym)
  {
    if (sym.base == B_STR && !sym.is_array()) {
      string value;
      if (string_value(int(&sym - syms.data()), value)) {
        return str_literal(value);
      }
    }
    if (sym.kind == SYM_CONST && !sym.is_array()) {
      return literal(sym, 0);
    }
    if (sym.alias != -1) {
      return syms[sym.alias].cname;
    }
    return sym.cname;
  }

  /* Read a scalar symbol as type `want`. */
  bool read(const int index, const VT want, string &result)
  {
    const Sym &sym = syms[index];
    if (sym.is_array()) {
      return fail("array " + sym.name + " used as a value");
    }
    if (sym.base == B_CLOSURE || sym.base == B_OTHER) {
      return fail("unsupported use of " + sym.type_name + " " + sym.name);
    }
    if ((sym.base == B_STR) != (want.b == B_STR)) {
      return fail("unsupported conversion of " + sym.type_name + " " + sym.name);
    }
    if (sym.base == B_STR && sym.str_top) {
      return fail("the string " + sym.name +
                  " is computed at render time in a way that is not supported on this device");
    }
    return convert(expr(sym), vt(sym), want, result);
  }

  /* Assign an expression of type `type` to a scalar symbol. */
  bool write(const int index, const string &value, const VT type, string &code)
  {
    const Sym &sym = syms[index];
    if (sym.is_array() || sym.kind == SYM_CONST) {
      return fail("unsupported assignment to " + sym.name);
    }
    if (sym.base == B_CLOSURE) {
      return true;
    }
    if (sym.base == B_OTHER || (sym.base == B_STR) != (type.b == B_STR)) {
      return fail("unsupported use of " + sym.type_name + " " + sym.name);
    }
    if (sym.base == B_STR && (sym.str_known || sym.str_top)) {
      /* A constant, or a string that is never looked at. */
      return true;
    }
    string converted;
    if (!convert(value, type, vt(sym), converted)) {
      return false;
    }
    code += sym.cname + " = " + converted + ";\n";
    return true;
  }

  bool string_arg(const Op &op, const int arg, string &value)
  {
    if (arg >= int(op.args.size()) || !string_value(op.args[arg], value)) {
      const string name = (arg < int(op.args.size())) ? syms[op.args[arg]].name : "?";
      return fail("instruction " + op.name + " needs a string that is known before rendering, but " +
                  name + " is not");
    }
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* Named coordinate systems */

  /* Matrix from a named space to common space, empty for the identity. */
  static string space_to_common(const string &space, bool &known)
  {
    known = true;
    if (space == "camera") {
      return "osl_kd_transform(lp, OSL_KD_CAM_CAMERATOWORLD)";
    }
    if (space == "screen") {
      return "osl_kd_projection(lp, OSL_KD_CAM_SCREENTOWORLD)";
    }
    if (space == "raster") {
      return "osl_kd_projection(lp, OSL_KD_CAM_RASTERTOWORLD)";
    }
    if (space == "NDC") {
      return "osl_kd_projection(lp, OSL_KD_CAM_NDCTOWORLD)";
    }
    known = (space == "common" || space == "world" || space == "object" || space == "shader");
    return "";
  }

  static string common_to_space(const string &space, bool &known)
  {
    known = true;
    if (space == "camera") {
      return "osl_kd_transform(lp, OSL_KD_CAM_WORLDTOCAMERA)";
    }
    if (space == "screen") {
      return "osl_kd_projection(lp, OSL_KD_CAM_WORLDTOSCREEN)";
    }
    if (space == "raster") {
      return "osl_kd_projection(lp, OSL_KD_CAM_WORLDTORASTER)";
    }
    if (space == "NDC") {
      return "osl_kd_projection(lp, OSL_KD_CAM_WORLDTONDC)";
    }
    known = (space == "common" || space == "world" || space == "object" || space == "shader");
    return "";
  }

  /* Matrix between two named spaces, empty for the identity. */
  static string space_to_space(const string &from, const string &to, bool &known)
  {
    bool from_known, to_known;
    const string a = space_to_common(from, from_known);
    const string b = common_to_space(to, to_known);
    known = from_known && to_known;
    if (from == to || (a.empty() && b.empty())) {
      return "";
    }
    if (a.empty()) {
      return b;
    }
    if (b.empty()) {
      return a;
    }
    return "osl_mat_mul(" + a + ", " + b + ")";
  }

  /* ------------------------------------------------------------------ */
  /* Colors */

  /* Rows of the matrix from CIE XYZ to the scene linear color space, and its luminance weights,
   * derived from the chromaticities of the primaries as the OSL runtime does. */
  void color_system(float xyz_to_rgb[3][3], float luminance[3]) const
  {
    /* Rec.709 primaries and D65 white point. */
    float xr = 0.64f, yr = 0.33f, xg = 0.30f, yg = 0.60f, xb = 0.15f, yb = 0.06f;
    float xw = 0.3127f, yw = 0.3291f;
    if (options.colorspace == "lin_rec2020_scene") {
      /* OSL names this "HDTV". */
      xr = 0.67f, yr = 0.33f, xg = 0.21f, yg = 0.71f, xb = 0.15f, yb = 0.06f;
    }
    else if (options.colorspace == "lin_ap1_scene") {
      xr = 0.713f, yr = 0.293f, xg = 0.165f, yg = 0.830f, xb = 0.128f, yb = 0.044f;
      xw = 0.32168f, yw = 0.33767f;
    }
    const float zr = 1.0f - (xr + yr), zg = 1.0f - (xg + yg), zb = 1.0f - (xb + yb);
    const float zw = 1.0f - (xw + yw);

    float rx = (yg * zb) - (yb * zg), ry = (xb * zg) - (xg * zb), rz = (xg * yb) - (xb * yg);
    float gx = (yb * zr) - (yr * zb), gy = (xr * zb) - (xb * zr), gz = (xb * yr) - (xr * yb);
    float bx = (yr * zg) - (yg * zr), by = (xg * zr) - (xr * zg), bz = (xr * yg) - (xg * yr);

    const float rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw;
    const float gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw;
    const float bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw;

    xyz_to_rgb[0][0] = rx / rw, xyz_to_rgb[0][1] = ry / rw, xyz_to_rgb[0][2] = rz / rw;
    xyz_to_rgb[1][0] = gx / gw, xyz_to_rgb[1][1] = gy / gw, xyz_to_rgb[1][2] = gz / gw;
    xyz_to_rgb[2][0] = bx / bw, xyz_to_rgb[2][1] = by / bw, xyz_to_rgb[2][2] = bz / bw;

    /* Luminance is the Y row of the inverse matrix. */
    const float (*m)[3] = xyz_to_rgb;
    const float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    luminance[0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    luminance[1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    luminance[2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
  }

  string xyz_to_rgb_expr(const string &c) const
  {
    float m[3][3], lum[3];
    color_system(m, lum);
    string s = "osl_color_matrix(" + c;
    for (int i = 0; i < 3; i++) {
      s += ", float3(" + float_literal(m[i][0]) + ", " + float_literal(m[i][1]) + ", " +
           float_literal(m[i][2]) + ")";
    }
    return s + ")";
  }

  string rgb_to_xyz_expr(const string &c) const
  {
    float m[3][3], lum[3];
    color_system(m, lum);
    /* Invert the matrix. */
    const float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    float inv[3][3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    string s = "osl_color_matrix(" + c;
    for (int i = 0; i < 3; i++) {
      s += ", float3(" + float_literal(inv[i][0]) + ", " + float_literal(inv[i][1]) + ", " +
           float_literal(inv[i][2]) + ")";
    }
    return s + ")";
  }

  bool is_rgb_space(const string &space) const
  {
    return space == "rgb" || space == "RGB" || space == "linear" || space == "scene_linear" ||
           space == options.colorspace ||
           (options.colorspace == "lin_rec709_scene" && space == "Rec709") ||
           (options.colorspace == "lin_rec2020_scene" && space == "HDTV") ||
           (options.colorspace == "lin_ap1_scene" && space == "ACEScg");
  }

  bool color_to_rgb(const Op &op, const string &space, const string &c, string &result)
  {
    if (is_rgb_space(space)) {
      result = c;
    }
    else if (space == "hsv") {
      result = "osl_hsv_to_rgb(" + c + ")";
    }
    else if (space == "hsl") {
      result = "osl_hsl_to_rgb(" + c + ")";
    }
    else if (space == "YIQ") {
      result = "osl_yiq_to_rgb(" + c + ")";
    }
    else if (space == "XYZ") {
      result = xyz_to_rgb_expr(c);
    }
    else if (space == "xyY") {
      result = xyz_to_rgb_expr("osl_xyy_to_xyz(" + c + ")");
    }
    else if (space == "sRGB") {
      result = "osl_srgb_to_linear(" + c + ")";
    }
    else {
      return fail("instruction " + op.name + " uses unsupported color space \"" + space + "\"");
    }
    return true;
  }

  bool color_from_rgb(const Op &op, const string &space, const string &c, string &result)
  {
    if (is_rgb_space(space)) {
      result = c;
    }
    else if (space == "hsv") {
      result = "osl_rgb_to_hsv(" + c + ")";
    }
    else if (space == "hsl") {
      result = "osl_rgb_to_hsl(" + c + ")";
    }
    else if (space == "YIQ") {
      result = "osl_rgb_to_yiq(" + c + ")";
    }
    else if (space == "XYZ") {
      result = rgb_to_xyz_expr(c);
    }
    else if (space == "xyY") {
      result = "osl_xyz_to_xyy(" + rgb_to_xyz_expr(c) + ")";
    }
    else if (space == "sRGB") {
      result = "osl_linear_to_srgb(" + c + ")";
    }
    else {
      return fail("instruction " + op.name + " uses unsupported color space \"" + space + "\"");
    }
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* Instructions */

  bool any_dual(const Op &op, const int first_arg = 1) const
  {
    for (int i = first_arg; i < int(op.args.size()); i++) {
      if (syms[op.args[i]].dual) {
        return true;
      }
    }
    return false;
  }

  /* Read all arguments after the result as type `t`, separated by commas. */
  bool read_args(const Op &op, const VT t, string &result, const int first_arg = 1)
  {
    result.clear();
    for (int i = first_arg; i < int(op.args.size()); i++) {
      string arg;
      if (!read(op.args[i], t, arg)) {
        return false;
      }
      result += ((i > first_arg) ? ", " : "") + arg;
    }
    return true;
  }

  /* An instruction that applies a function to each component: all arguments convert to the
   * type of the result. */
  bool emit_function(const Op &op, const string &function, const bool has_derivs, string &code)
  {
    const Sym &dst = syms[op.args[0]];
    VT t = {dst.base, false};
    if (t.b == B_INT) {
      /* Integer results of float functions convert on assignment. */
      for (size_t i = 1; i < op.args.size(); i++) {
        if (syms[op.args[i]].base != B_INT) {
          t.b = B_FLOAT;
        }
      }
    }
    if (t.b != B_INT && t.b != B_FLOAT && t.b != B_VEC) {
      return fail("instruction " + op.name + " is not supported for " + dst.type_name);
    }
    t.d = has_derivs && dst.dual && any_dual(op);
    string args;
    if (!read_args(op, t, args)) {
      return false;
    }
    return write(op.args[0], function + "(" + args + ")", t, code);
  }

  bool emit_arithmetic(const Op &op, string &code)
  {
    const Sym &dst = syms[op.args[0]];
    const string &name = op.name;
    const int nargs = int(op.args.size()) - 1;

    if (dst.base == B_MAT) {
      if (name == "neg" && nargs == 1) {
        string value;
        if (!read(op.args[1], {B_MAT, false}, value)) {
          return false;
        }
        return write(op.args[0], "osl_mat_neg(" + value + ")", {B_MAT, false}, code);
      }
      if (nargs != 2 || (name != "mul" && name != "div")) {
        return fail("instruction " + name + " is not supported for matrices");
      }
      const Sym &a = syms[op.args[1]];
      const Sym &b = syms[op.args[2]];
      string ea, eb;
      if (!read(op.args[1], {a.base == B_MAT ? B_MAT : B_FLOAT, false}, ea) ||
          !read(op.args[2], {b.base == B_MAT ? B_MAT : B_FLOAT, false}, eb))
      {
        return false;
      }
      string value;
      if (name == "mul") {
        if (a.base == B_MAT && b.base == B_MAT) {
          value = "osl_mat_mul(" + ea + ", " + eb + ")";
        }
        else if (a.base == B_MAT) {
          value = "osl_mat_scale(" + ea + ", " + eb + ")";
        }
        else {
          value = "osl_mat_scale(" + eb + ", " + ea + ")";
        }
      }
      else {
        if (a.base == B_MAT && b.base == B_MAT) {
          value = "osl_mat_mul(" + ea + ", osl_mat_inverse(" + eb + "))";
        }
        else if (a.base == B_MAT) {
          value = "osl_mat_scale(" + ea + ", osl_div(1.0f, " + eb + "))";
        }
        else {
          value = "osl_mat_scale(osl_mat_inverse(" + eb + "), " + ea + ")";
        }
      }
      return write(op.args[0], value, {B_MAT, false}, code);
    }

    VT t = {dst.base, false};
    if (t.b == B_INT) {
      for (size_t i = 1; i < op.args.size(); i++) {
        if (syms[op.args[i]].base != B_INT) {
          t.b = B_FLOAT;
        }
      }
    }
    if (t.b != B_INT && t.b != B_FLOAT && t.b != B_VEC) {
      return fail("instruction " + name + " is not supported for " + dst.type_name);
    }
    t.d = dst.dual && any_dual(op);

    vector<string> a(nargs);
    for (int i = 0; i < nargs; i++) {
      if (!read(op.args[i + 1], t, a[i])) {
        return false;
      }
    }

    string value;
    if (name == "neg" && nargs == 1) {
      value = t.d ? "osl_neg(" + a[0] + ")" : "(-" + a[0] + ")";
    }
    else if (nargs != 2) {
      return fail("malformed instruction " + name);
    }
    else if (name == "div") {
      value = "osl_div(" + a[0] + ", " + a[1] + ")";
    }
    else if (t.d) {
      value = "osl_" + name + "(" + a[0] + ", " + a[1] + ")";
    }
    else {
      const char *symbol = (name == "add") ? " + " : (name == "sub") ? " - " : " * ";
      value = "(" + a[0] + symbol + a[1] + ")";
    }
    return write(op.args[0], value, t, code);
  }

  bool emit_compare(const Op &op, string &code)
  {
    if (op.args.size() != 3) {
      return fail("malformed instruction " + op.name);
    }
    const Sym &a = syms[op.args[1]];
    const Sym &b = syms[op.args[2]];
    const string &name = op.name;
    string value;

    if (a.base == B_STR || b.base == B_STR) {
      if (name != "eq" && name != "neq") {
        return fail("instruction " + name + " is not supported for strings");
      }
      string sa, sb;
      if (string_value(op.args[1], sa) && string_value(op.args[2], sb)) {
        value = ((sa == sb) == (name == "eq")) ? "1" : "0";
      }
      else {
        /* Equal strings have the same index in the table. */
        string ea, eb;
        if (!read(op.args[1], {B_STR, false}, ea) || !read(op.args[2], {B_STR, false}, eb)) {
          return false;
        }
        value = "int(" + ea + ((name == "eq") ? " == " : " != ") + eb + ")";
      }
    }
    else if (a.base == B_MAT || b.base == B_MAT) {
      string ea, eb;
      if (!read(op.args[1], {B_MAT, false}, ea) || !read(op.args[2], {B_MAT, false}, eb)) {
        return false;
      }
      if (name == "eq") {
        value = "osl_mat_eq(" + ea + ", " + eb + ")";
      }
      else if (name == "neq") {
        value = "(1 - osl_mat_eq(" + ea + ", " + eb + "))";
      }
      else {
        return fail("instruction " + name + " is not supported for matrices");
      }
    }
    else {
      VT t = {B_FLOAT, false};
      if (a.base == B_VEC || b.base == B_VEC) {
        t.b = B_VEC;
      }
      else if (a.base == B_INT && b.base == B_INT) {
        t.b = B_INT;
      }
      string ea, eb;
      if (!read(op.args[1], t, ea) || !read(op.args[2], t, eb)) {
        return false;
      }
      const char *symbol = (name == "eq")  ? " == " :
                           (name == "neq") ? " != " :
                           (name == "lt")  ? " < " :
                           (name == "le")  ? " <= " :
                           (name == "gt")  ? " > " :
                                             " >= ";
      if (t.b == B_VEC) {
        if (name == "eq") {
          value = "int(all(" + ea + " == " + eb + "))";
        }
        else if (name == "neq") {
          value = "int(any(" + ea + " != " + eb + "))";
        }
        else {
          return fail("instruction " + name + " is not supported for vectors");
        }
      }
      else {
        value = "int(" + ea + symbol + eb + ")";
      }
    }
    return write(op.args[0], value, {B_INT, false}, code);
  }

  bool emit_integer(const Op &op, string &code)
  {
    const string &name = op.name;
    string args;
    vector<string> a(op.args.size());
    for (size_t i = 1; i < op.args.size(); i++) {
      if (!read(op.args[i], {B_INT, false}, a[i])) {
        return false;
      }
    }
    string value;
    if (name == "compl" && op.args.size() == 2) {
      value = "(~" + a[1] + ")";
    }
    else if (name == "not" && op.args.size() == 2) {
      value = "int(" + a[1] + " == 0)";
    }
    else if (op.args.size() != 3) {
      return fail("malformed instruction " + name);
    }
    else if (name == "and") {
      value = "int((" + a[1] + " != 0) && (" + a[2] + " != 0))";
    }
    else if (name == "or") {
      value = "int((" + a[1] + " != 0) || (" + a[2] + " != 0))";
    }
    else if (name == "shl" || name == "shr") {
      value = "(" + a[1] + ((name == "shl") ? " << " : " >> ") + "(" + a[2] + " & 31))";
    }
    else {
      const char *symbol = (name == "bitand") ? " & " : (name == "bitor") ? " | " : " ^ ";
      value = "(" + a[1] + symbol + a[2] + ")";
    }
    return write(op.args[0], value, {B_INT, false}, code);
  }

  /* Element of an array symbol. */
  string element(const Sym &array, const string &index)
  {
    return expr(array) + "[osl_idx(" + index + ", " + format("%d", array.arraylen) + ")]";
  }

  bool emit_assign(const Op &op, string &code)
  {
    if (op.args.size() != 2) {
      return fail("malformed instruction " + op.name);
    }
    const Sym &dst = syms[op.args[0]];
    const Sym &src = syms[op.args[1]];
    if (dst.base == B_STR && dst.str_top) {
      /* Never looked at. */
      return true;
    }
    if (dst.is_array()) {
      if (dst.alias != -1) {
        return true;
      }
      if (!src.is_array() || dst.arraylen <= 0 || src.arraylen <= 0) {
        return fail("unsupported array assignment to " + dst.name);
      }
      string value;
      if (!convert(expr(src) + "[i]", vt(src), vt(dst), value)) {
        return false;
      }
      code += format("for (int i = 0; i < %d; i++) {\n", std::min(dst.arraylen, src.arraylen));
      code += dst.cname + "[i] = " + value + ";\n}\n";
      return true;
    }
    string value;
    if (!read(op.args[1], vt(src), value)) {
      return false;
    }
    return write(op.args[0], value, vt(src), code);
  }

  bool emit_getattribute(const Op &op, string &code)
  {
    /* getattribute([object,] name, [index,] value) */
    const int nargs = int(op.args.size());
    if (nargs < 3) {
      return fail("malformed instruction getattribute");
    }
    const int value_arg = nargs - 1;
    int name_arg = 1;
    bool indexed = false;
    if (nargs == 5) {
      name_arg = 2;
      indexed = true;
    }
    else if (nargs == 4) {
      if (syms[op.args[2]].base == B_STR) {
        name_arg = 2;
      }
      else {
        indexed = true;
      }
    }
    string name;
    if (!string_arg(op, name_arg, name)) {
      return false;
    }

    /* Attributes are one or two floats. */
    string x, y;
    if (name == "cam:sensor_size") {
      x = "osl_kd(lp, OSL_KD_CAM_SENSORWIDTH)";
      y = "osl_kd(lp, OSL_KD_CAM_SENSORHEIGHT)";
    }
    else if (name == "cam:image_resolution") {
      x = "osl_kd(lp, OSL_KD_CAM_WIDTH)";
      y = "osl_kd(lp, OSL_KD_CAM_HEIGHT)";
    }
    else if (name == "cam:aperture_aspect_ratio") {
      x = "(1.0f / osl_kd(lp, OSL_KD_CAM_INV_APERTURE_RATIO))";
    }
    else if (name == "cam:aperture_size") {
      x = "osl_kd(lp, OSL_KD_CAM_APERTURESIZE)";
    }
    else if (name == "cam:aperture_position") {
      x = "in[11]";
      y = "in[12]";
    }
    else if (name == "cam:focal_distance") {
      x = "osl_kd(lp, OSL_KD_CAM_FOCALDISTANCE)";
    }
    else if (name == "scene:time") {
      x = "osl_kd(lp, OSL_KD_SCENE_TIME)";
    }
    else if (name == "scene:frame") {
      x = "osl_kd(lp, OSL_KD_SCENE_FRAME)";
    }

    const Sym &value = syms[op.args[value_arg]];
    bool found = !indexed && !x.empty();
    if (found) {
      /* Same conversions as the attributes of other devices: a float fills all components, a
       * pair of floats is padded with zero or averaged. */
      if (value.base == B_FLOAT && !value.is_array()) {
        const string f = y.empty() ? x : "((" + x + " + " + y + ") * 0.5f)";
        if (!write(op.args[value_arg], f, {B_FLOAT, false}, code)) {
          return false;
        }
      }
      else if (value.base == B_VEC && !value.is_array()) {
        const string v = y.empty() ? "float3(" + x + ")" : "float3(" + x + ", " + y + ", 0.0f)";
        if (!write(op.args[value_arg], v, {B_VEC, false}, code)) {
          return false;
        }
      }
      else if (value.base == B_FLOAT && (value.arraylen == 3 || value.arraylen == 4) &&
               value.alias == -1 && value.kind != SYM_CONST)
      {
        const string c[4] = {x, y.empty() ? x : y, y.empty() ? x : "0.0f", "1.0f"};
        for (int i = 0; i < value.arraylen; i++) {
          string converted;
          if (!convert(c[i], {B_FLOAT, false}, vt(value), converted)) {
            return false;
          }
          code += value.cname + format("[%d] = ", i) + converted + ";\n";
        }
      }
      else {
        found = false;
      }
    }
    return write(op.args[0], found ? "1" : "0", {B_INT, false}, code);
  }

  /* Which parts of the runtime library the shader needs. */
  bool use_perlin = false;
  bool use_periodic = false;
  bool use_simplex = false;
  bool use_gabor = false;
  bool use_spline = false;
  bool use_color_derivs = false;

  bool emit_noise(const Op &op, string &code)
  {
    const int nargs = int(op.args.size());
    const bool periodic = (op.name == "pnoise" || op.name == "psnoise");
    const bool generic = (op.name == "noise" || op.name == "pnoise");
    const Sym &dst = syms[op.args[0]];
    if (dst.base != B_FLOAT && dst.base != B_VEC) {
      return fail("malformed instruction " + op.name);
    }
    const auto zero_result = [&](const float value) {
      return write(op.args[0],
                   (dst.base == B_VEC) ? "float3(" + float_literal(value) + ")" :
                                         float_literal(value),
                   {dst.base, false},
                   code);
    };

    /* The kind of noise: the instruction itself, or a name as the first argument. */
    string function = (op.name == "noise")   ? "noise" :
                      (op.name == "snoise")  ? "snoise" :
                      (op.name == "pnoise")  ? "pnoise" :
                      (op.name == "psnoise") ? "psnoise" :
                                               op.name;
    int first = 1;
    bool has_derivs = (op.name != "cellnoise" && op.name != "hashnoise");
    bool gabor = false;
    if (generic && nargs > 1 && syms[op.args[1]].base == B_STR) {
      string type;
      if (!string_arg(op, 1, type)) {
        return false;
      }
      first = 2;
      if (type == "uperlin" || type == "noise") {
        function = periodic ? "pnoise" : "noise";
      }
      else if (type == "perlin" || type == "snoise") {
        function = periodic ? "psnoise" : "snoise";
      }
      else if (type == "cell") {
        function = periodic ? "pcellnoise" : "cellnoise";
        has_derivs = false;
      }
      else if (type == "hash") {
        function = periodic ? "phashnoise" : "hashnoise";
        has_derivs = false;
      }
      else if (type == "gabor") {
        function = periodic ? "pgabor" : "gabor";
        gabor = true;
      }
      else if (!periodic && type == "simplex") {
        function = "simplexnoise";
      }
      else if (!periodic && type == "usimplex") {
        function = "usimplexnoise";
      }
      else if (!str_fixed.count(op.args[1])) {
        /* The OSL runtime knows fewer names for a constant than for a name chosen at render
         * time, and fails to compile the shader. */
        if (type == "cellnoise") {
          function = periodic ? "pcellnoise" : "cellnoise";
          has_derivs = false;
        }
        else if (type == "hashnoise") {
          function = periodic ? "phashnoise" : "hashnoise";
          has_derivs = false;
        }
        else {
          return fail("noise type \"" + type + "\" is unknown");
        }
      }
      else if (!periodic && type == "simplexnoise") {
        function = "simplexnoise";
      }
      else if (!periodic && type == "usimplexnoise") {
        function = "usimplexnoise";
      }
      else if (!periodic && type == "null") {
        return zero_result(0.0f);
      }
      else if (!periodic && type == "unull") {
        return zero_result(0.5f);
      }
      else {
        /* Unknown kinds of noise are an error message and no result in the OSL runtime. */
        return true;
      }
    }

    /* Coordinates, periods, then options as pairs of a name and a value. */
    int num_positional = 0;
    while (first + num_positional < nargs && syms[op.args[first + num_positional]].base != B_STR)
    {
      num_positional++;
    }
    const int num_coords = periodic ? num_positional / 2 : num_positional;
    if (num_coords < 1 || num_coords > 2 || (periodic && num_positional != num_coords * 2) ||
        ((nargs - first - num_positional) & 1))
    {
      return fail("malformed instruction " + op.name);
    }
    const Sym &p = syms[op.args[first]];
    if (p.base != B_FLOAT && p.base != B_VEC && p.base != B_INT) {
      return fail("malformed instruction " + op.name);
    }
    const bool vector_p = (p.base == B_VEC);
    const int dimensions = (vector_p ? 3 : 1) + (num_coords - 1);

    bool dual = has_derivs && dst.dual;
    if (dual) {
      dual = false;
      for (int i = 0; i < num_coords; i++) {
        dual |= syms[op.args[first + i]].dual;
      }
    }
    /* Gabor noise filters with the derivatives of the position. */
    const bool dual_args = gabor || dual;

    string args;
    for (int i = 0; i < num_coords; i++) {
      string arg;
      if (!read(op.args[first + i], {(i == 0 && vector_p) ? B_VEC : B_FLOAT, dual_args}, arg)) {
        return false;
      }
      args += ((i > 0) ? ", " : "") + arg;
    }
    for (int i = 0; i < (periodic ? num_coords : 0); i++) {
      string arg;
      if (!read(op.args[first + num_coords + i], {(i == 0 && vector_p) ? B_VEC : B_FLOAT, false},
                arg))
      {
        return false;
      }
      args += ", " + arg;
    }

    if (gabor) {
      /* Options. */
      string anisotropic = "0", direction = "float3(1.0f, 0.0f, 0.0f)", bandwidth = "1.0f";
      string impulses = "16.0f", do_filter = "1";
      for (int i = first + num_positional; i + 1 < nargs; i += 2) {
        string option;
        if (!string_arg(op, i, option)) {
          return false;
        }
        if (option == "anisotropic" || option == "do_filter") {
          if (!read(op.args[i + 1], {B_INT, false},
                    (option == "anisotropic") ? anisotropic : do_filter))
          {
            return false;
          }
        }
        else if (option == "direction") {
          if (!read(op.args[i + 1], {B_VEC, false}, direction)) {
            return false;
          }
        }
        else if (option == "bandwidth" || option == "impulses") {
          if (!read(op.args[i + 1], {B_FLOAT, false}, (option == "bandwidth") ? bandwidth : impulses))
          {
            return false;
          }
        }
      }
      use_gabor = true;
      args += ", osl_gabor_params(" + anisotropic + ", " + direction + ", " + bandwidth + ", " +
              impulses + ", " + do_filter + ")";
      const string call = "osl_" + function + ((dst.base == B_VEC) ? "_v" : "_f") +
                          format("%d", dimensions) + "(" + args + ")";
      return write(op.args[0], call, {dst.base, true}, code);
    }

    if (function == "simplexnoise" || function == "usimplexnoise") {
      use_simplex = true;
      use_perlin = true;
    }
    else if (function != "cellnoise" && function != "hashnoise") {
      use_perlin = true;
    }
    if (periodic) {
      use_periodic = true;
      use_perlin = true;
    }
    const string call = "osl_" + function + ((dst.base == B_VEC) ? "_v" : "_f") +
                        format("%d", dimensions) + "(" + args + ")";
    return write(op.args[0], call, {dst.base, dual}, code);
  }

  /* Images. The function cannot look them up: it returns a request to the kernel, which calls
   * the function again with the result. Every lookup therefore first checks if its result is
   * among the results passed in, in the order of the lookups. */
  bool use_images = false;
  int num_image_ops = 0;
  vector<OSLCameraTranslateResult::Image> images;

  /* Parameter word holding the kernel ID of an image. */
  int image_slot(const string &filename)
  {
    for (const OSLCameraTranslateResult::Image &image : images) {
      if (image.filename == filename) {
        return image.offset;
      }
    }
    images.push_back({filename, num_words});
    return num_words++;
  }

  /* Code that gets the result of a lookup into `float tex_r[5]`, or returns the request.
   * `request` are the values of the request after the type and the image. */
  string image_lookup(const int type, const string &filename, const vector<string> &request)
  {
    use_images = true;
    string code = "float tex_r[5];\nif (tex_next < tex_avail) {\nfor (int i = 0; i < 5; i++) {\n"
                  "tex_r[i] = in[14 + 5 * tex_next + i];\n}\ntex_next++;\n}\nelse {\n";
    code += format("out[21] = %d.0f;\nout[22] = as_type<float>(prm[%d]);\n",
                   type,
                   image_slot(filename));
    for (size_t i = 0; i < request.size(); i++) {
      if (!request[i].empty()) {
        code += format("out[%d] = ", int(23 + i)) + request[i] + ";\n";
      }
    }
    code += "return true;\n}\n";
    return code;
  }

  bool emit_texture(const Op &op, string &code)
  {
    /* texture result filename s t [dsdx dtdx dsdy dtdy] options...
     * texture3d result filename p [dpdx dpdy dpdz] options...
     * environment result filename r [drdx drdy] options... */
    const int nargs = int(op.args.size());
    const string &name = op.name;
    string filename;
    if (nargs < 3 || !string_arg(op, 1, filename)) {
      return false;
    }
    const Sym &dst = syms[op.args[0]];
    if (dst.base != B_FLOAT && dst.base != B_VEC) {
      return fail("malformed instruction " + name);
    }
    int num_positional = 0;
    while (2 + num_positional < nargs && syms[op.args[2 + num_positional]].base != B_STR) {
      num_positional++;
    }
    if ((nargs - 2 - num_positional) & 1) {
      return fail("malformed instruction " + name);
    }

    /* The request: coordinates and the color for missing images. */
    vector<string> request(12, "0.0f");
    int type;
    if (name == "texture") {
      type = 1;
      if (num_positional != 2 && num_positional != 6) {
        return fail("malformed instruction texture");
      }
      if (num_positional == 6) {
        for (int i = 0; i < 6; i++) {
          if (!read(op.args[2 + i], {B_FLOAT, false}, request[i])) {
            return false;
          }
        }
      }
      else {
        /* The derivatives of the coordinates select the resolution. */
        for (int i = 0; i < 2; i++) {
          const Sym &coord = syms[op.args[2 + i]];
          string value;
          if (!read(op.args[2 + i], {B_FLOAT, coord.dual}, value)) {
            return false;
          }
          request[i] = coord.dual ? "(" + value + ").v" : value;
          if (coord.dual) {
            request[2 + i] = "(" + value + ").dx";
            request[4 + i] = "(" + value + ").dy";
          }
        }
      }
    }
    else {
      type = (name == "texture3d") ? 2 : 3;
      if (num_positional < 1) {
        return fail("malformed instruction " + name);
      }
      string p;
      if (!read(op.args[2], {B_VEC, false}, p)) {
        return false;
      }
      code += "{\nconst float3 tex_p = " + p + ";\n";
      request[0] = "tex_p.x";
      request[1] = "tex_p.y";
      request[2] = "tex_p.z";
    }

    /* Options. Only the color of missing images affects the lookup, as on other devices. */
    string alpha_code;
    for (int i = 2 + num_positional; i + 1 < nargs; i += 2) {
      string option;
      if (!string_arg(op, i, option)) {
        return false;
      }
      const Sym &value = syms[op.args[i + 1]];
      if (option == "missingcolor" && value.base == B_VEC) {
        string color;
        if (!read(op.args[i + 1], {B_VEC, false}, color)) {
          return false;
        }
        request[6] = "1.0f";
        request[7] = "(" + color + ").x";
        request[8] = "(" + color + ").y";
        request[9] = "(" + color + ").z";
      }
      else if (option == "missingalpha" && (value.base == B_FLOAT || value.base == B_INT)) {
        request[6] = "1.0f";
        if (!read(op.args[i + 1], {B_FLOAT, false}, request[10])) {
          return false;
        }
      }
      else if (option == "alpha" && value.base == B_FLOAT && op.writes(i + 1)) {
        /* The channel after the ones of the result. */
        if (!write(op.args[i + 1],
                   (dst.base == B_VEC) ? "tex_r[3]" : "tex_r[1]",
                   {B_FLOAT, false},
                   alpha_code))
        {
          return false;
        }
      }
      else if (option == "errormessage" && value.base == B_STR && op.writes(i + 1)) {
        if (!write(op.args[i + 1], str_literal(""), {B_STR, false}, alpha_code)) {
          return false;
        }
      }
    }

    string block = image_lookup(type, filename, request);
    if (!write(op.args[0],
               (dst.base == B_VEC) ? "float3(tex_r[0], tex_r[1], tex_r[2])" : "tex_r[0]",
               {dst.base, false},
               block))
    {
      return false;
    }
    block += alpha_code;
    code += "{\n" + block + "}\n";
    if (type != 1) {
      code += "}\n";
    }
    return true;
  }

  bool emit_gettextureinfo(const Op &op, string &code)
  {
    /* gettextureinfo result filename [s t] dataname data */
    const int nargs = int(op.args.size());
    string filename, dataname;
    if ((nargs != 4 && nargs != 6) || !string_arg(op, 1, filename) ||
        !string_arg(op, nargs - 2, dataname))
    {
      return fail("malformed instruction gettextureinfo");
    }
    const Sym &data = syms[op.args[nargs - 1]];
    const bool is_float3 = (data.base == B_VEC && !data.is_array()) ||
                           (data.base == B_FLOAT && data.arraylen == 3);
    /* What is known about images, and the types it can be stored in. */
    int info = 0;
    int count = 0;
    if (dataname == "resolution") {
      if ((data.base == B_INT || data.base == B_FLOAT) && data.arraylen == 2) {
        info = 1;
        count = 2;
      }
      else if ((data.base == B_INT && data.arraylen == 3) || is_float3) {
        info = 1;
        count = 3;
      }
    }
    else if (dataname == "channels" || dataname == "exists") {
      if (data.base == B_INT && !data.is_array()) {
        info = (dataname == "channels") ? 2 : 3;
        count = 1;
      }
    }
    else if (dataname == "averagecolor") {
      if (is_float3) {
        info = 4;
        count = 3;
      }
      else if (data.base == B_FLOAT && data.arraylen == 4) {
        info = 4;
        count = 4;
      }
    }
    if (info == 0 || data.kind == SYM_CONST || data.alias != -1 || data.promoted) {
      return write(op.args[0], "0", {B_INT, false}, code);
    }

    vector<string> request(12, "0.0f");
    if (nargs == 6) {
      if (!read(op.args[2], {B_FLOAT, false}, request[0]) ||
          !read(op.args[3], {B_FLOAT, false}, request[1]))
      {
        return false;
      }
    }
    request[11] = format("%d.0f", info);
    string block = image_lookup((nargs == 6) ? 5 : 4, filename, request);

    string found;
    if (data.is_array()) {
      for (int i = 0; i < count; i++) {
        string value;
        if (!convert(format("tex_r[%d]", i), {B_FLOAT, false}, vt(data), value)) {
          return false;
        }
        found += data.cname + format("[%d] = ", i) + value + ";\n";
      }
    }
    else if (data.base == B_VEC) {
      if (!write(op.args[nargs - 1], "float3(tex_r[0], tex_r[1], tex_r[2])", {B_VEC, false}, found))
      {
        return false;
      }
    }
    else if (!write(op.args[nargs - 1], "tex_r[0]", {B_FLOAT, false}, found)) {
      return false;
    }
    string found_result, missing_result;
    if (!write(op.args[0], "1", {B_INT, false}, found_result) ||
        !write(op.args[0], "0", {B_INT, false}, missing_result))
    {
      return false;
    }
    block += "if (tex_r[4] != 0.0f) {\n" + found + found_result + "}\nelse {\n" +
             missing_result + "}\n";
    code += "{\n" + block + "}\n";
    return true;
  }

  /* Index of a spline basis in the tables of the runtime library. */
  static int spline_basis(const string &name)
  {
    static const char *names[] = {"catmull-rom", "bezier", "bspline", "hermite", "linear",
                                  "constant"};
    for (int i = 0; i < 6; i++) {
      if (name == names[i]) {
        return i;
      }
    }
    /* Unknown bases are linear. */
    return 4;
  }

  bool emit_spline(const Op &op, string &code)
  {
    /* spline result basis x [count] knots */
    const int nargs = int(op.args.size());
    string basis_name;
    if ((nargs != 4 && nargs != 5) || !string_arg(op, 1, basis_name)) {
      return fail("malformed instruction " + op.name);
    }
    const int basis = spline_basis(basis_name);
    const Sym &dst = syms[op.args[0]];
    const Sym &x = syms[op.args[2]];
    const Sym &knots = syms[op.args[nargs - 1]];
    if (!knots.is_array() || knots.arraylen <= 0 || (knots.base != B_FLOAT && knots.base != B_VEC))
    {
      return fail("unsupported array " + knots.name);
    }
    const Sym &source = (knots.alias != -1) ? syms[knots.alias] : knots;
    string count = format("%d", knots.arraylen);
    if (nargs == 5 && !read(op.args[3], {B_INT, false}, count)) {
      return false;
    }
    use_spline = true;
    const string b = format("%d", basis);

    if (op.name == "splineinverse") {
      if (knots.base != B_FLOAT) {
        return fail("instruction splineinverse is not supported for " + knots.type_name);
      }
      string y, k;
      if (!read(op.args[2], {B_FLOAT, false}, y) ||
          !convert(expr(knots) + "[i]", vt(source), {B_FLOAT, false}, k))
      {
        return false;
      }
      string block = format("float spline_knots[%d];\n", knots.arraylen) +
                     format("for (int i = 0; i < %d; i++) {\n", knots.arraylen) +
                     "spline_knots[i] = " + k + ";\n}\n";
      if (!write(op.args[0],
                 "osl_splineinverse(" + b + ", " + y + ", spline_knots, " + count +
                     format(", %d)", knots.arraylen),
                 {B_FLOAT, false},
                 block))
      {
        return false;
      }
      code += "{\n" + block + "}\n";
      return true;
    }

    const bool dual = dst.dual && (x.dual || source.dual);
    const VT kt = {knots.base, dual};
    string xv;
    if (!read(op.args[2], {B_FLOAT, dual}, xv)) {
      return false;
    }
    string block = "const int spline_n = osl_spline_segments(" + b + ", " + count + ");\n" +
                   "const " + ctype({B_FLOAT, dual}) + " spline_x = " + xv + ";\n" +
                   "const int spline_seg = osl_spline_segment(osl_val(spline_x), spline_n);\n";
    const auto knot = [&](const string &index, const VT want, string &result) {
      return convert(element(knots, index), vt(source), want, result);
    };
    if (basis == 5) {
      string value;
      if (!knot("spline_seg + 1", {knots.base, false}, value) ||
          !write(op.args[0], value, {knots.base, false}, block))
      {
        return false;
      }
    }
    else {
      block += "const int spline_s = spline_seg * osl_spline_step[" + b + "];\n";
      string k[4];
      for (int i = 0; i < 4; i++) {
        if (!knot(format("spline_s + %d", i), kt, k[i])) {
          return false;
        }
      }
      if (!write(op.args[0],
                 "osl_spline_eval(" + b + ", osl_spline_param(spline_x, spline_n, spline_seg), " +
                     k[0] + ", " + k[1] + ", " + k[2] + ", " + k[3] + ")",
                 kt,
                 block))
      {
        return false;
      }
    }
    code += "{\n" + block + "}\n";
    return true;
  }

  bool emit_transform(const Op &op, string &code)
  {
    const string kind = (op.name == "transform")  ? "point" :
                        (op.name == "transformv") ? "vector" :
                                                    "normal";
    const int nargs = int(op.args.size());
    const int p_arg = nargs - 1;
    string matrix;
    if (nargs == 3) {
      const Sym &m = syms[op.args[1]];
      if (m.base == B_STR) {
        string to;
        bool known;
        if (!string_arg(op, 1, to)) {
          return false;
        }
        matrix = space_to_space("common", to, known);
      }
      else if (!read(op.args[1], {B_MAT, false}, matrix)) {
        return false;
      }
    }
    else if (nargs == 4) {
      string from, to;
      bool known;
      if (!string_arg(op, 1, from) || !string_arg(op, 2, to)) {
        return false;
      }
      matrix = space_to_space(from, to, known);
    }
    else {
      return fail("malformed instruction " + op.name);
    }

    const VT t = {B_VEC, syms[op.args[0]].dual && syms[op.args[p_arg]].dual};
    string p;
    if (!read(op.args[p_arg], t, p)) {
      return false;
    }
    const string value = matrix.empty() ? p :
                                          "osl_transform_" + kind + "(" + matrix + ", " + p + ")";
    return write(op.args[0], value, t, code);
  }

  bool emit_matrix(const Op &op, string &code)
  {
    const int nargs = int(op.args.size()) - 1;
    int first = 1;
    string space;
    bool has_space = false;
    if (nargs >= 1 && syms[op.args[1]].base == B_STR) {
      if (!string_arg(op, 1, space)) {
        return false;
      }
      has_space = true;
      first = 2;
    }
    const int nvalues = int(op.args.size()) - first;
    string value;
    bool known;
    if (has_space && nvalues == 1 && syms[op.args[first]].base == B_STR) {
      string to;
      if (!string_arg(op, first, to)) {
        return false;
      }
      value = space_to_space(space, to, known);
      if (value.empty()) {
        value = "osl_mat_diag(1.0f)";
      }
      return write(op.args[0], value, {B_MAT, false}, code);
    }
    string args;
    if (!read_args(op, {B_FLOAT, false}, args, first)) {
      return false;
    }
    if (nvalues == 16) {
      value = "osl_mat(" + args + ")";
    }
    else if (nvalues == 1) {
      value = "osl_mat_diag(" + args + ")";
    }
    else {
      return fail("malformed instruction matrix");
    }
    if (has_space) {
      const string to_common = space_to_common(space, known);
      if (!to_common.empty()) {
        value = "osl_mat_mul(" + to_common + ", " + value + ")";
      }
    }
    return write(op.args[0], value, {B_MAT, false}, code);
  }

  bool emit_construct(const Op &op, string &code)
  {
    /* color, point, vector or normal from components, optionally in a named space. */
    const Sym &dst = syms[op.args[0]];
    int first = 1;
    string space;
    bool has_space = false;
    if (op.args.size() > 1 && syms[op.args[1]].base == B_STR) {
      if (!string_arg(op, 1, space)) {
        return false;
      }
      has_space = true;
      first = 2;
    }
    const int nvalues = int(op.args.size()) - first;
    if (nvalues != 3 && nvalues != 1) {
      return fail("malformed instruction " + op.name);
    }

    /* The OSL runtime has no derivatives for colors given in a color space. */
    VT t = {B_FLOAT, dst.dual && any_dual(op, first) && !(op.name == "color" && has_space)};
    string args;
    if (!read_args(op, t, args, first)) {
      return false;
    }
    string value;
    if (nvalues == 3) {
      value = "osl_vec(" + args + ")";
    }
    else if (!convert(args, t, {B_VEC, t.d}, value)) {
      return false;
    }
    t.b = B_VEC;

    if (has_space) {
      if (op.name == "color") {
        if (!color_to_rgb(op, space, value, value)) {
          return false;
        }
      }
      else {
        bool known;
        const string matrix = space_to_common(space, known);
        if (!matrix.empty()) {
          const char *kind = (op.name == "point")  ? "point" :
                             (op.name == "vector") ? "vector" :
                                                     "normal";
          value = string("osl_transform_") + kind + "(" + matrix + ", " + value + ")";
        }
      }
    }
    return write(op.args[0], value, t, code);
  }

  /* An instruction with arguments of fixed types. `signature` gives the type of the result
   * followed by the types of the arguments: i, f, v or m. */
  bool emit_fixed(const Op &op,
                  const string &function,
                  const char *signature,
                  const bool has_derivs,
                  string &code)
  {
    if (op.args.size() != strlen(signature)) {
      return fail("malformed instruction " + op.name);
    }
    const auto base = [](const char c) {
      return (c == 'i') ? B_INT : (c == 'f') ? B_FLOAT : (c == 'v') ? B_VEC : B_MAT;
    };
    bool dual = has_derivs && syms[op.args[0]].dual;
    if (dual) {
      dual = false;
      for (size_t i = 1; i < op.args.size(); i++) {
        dual |= syms[op.args[i]].dual && (signature[i] == 'f' || signature[i] == 'v');
      }
    }
    string args;
    for (size_t i = 1; i < op.args.size(); i++) {
      const Base b = base(signature[i]);
      string arg;
      if (!read(op.args[i], {b, dual && (b == B_FLOAT || b == B_VEC)}, arg)) {
        return false;
      }
      args += ((i > 1) ? ", " : "") + arg;
    }
    const Base result = base(signature[0]);
    return write(op.args[0],
                 function + "(" + args + ")",
                 {result, dual && (result == B_FLOAT || result == B_VEC)},
                 code);
  }

  /* The string arguments whose text an instruction needs, as opposed to instructions that
   * only move strings around or compare them. */
  vector<int> string_content_args(const Op &op) const
  {
    static const std::set<string> no_content = {
        "assign",  "arraycopy", "aref",    "aassign",           "eq",
        "neq",     "printf",    "fprintf", "error",             "warning",
        "nop",     "end",       "useparam", "closure",          "trace",
        "pointcloud_search",    "pointcloud_get",               "pointcloud_write",
        "arraylength",
    };
    vector<int> args;
    if (no_content.count(op.name) || op.args.empty()) {
      return args;
    }
    const Sym &dst = syms[op.args[0]];
    if (is_string_function_op(op.name) && (dst.str_top || dst.str_known)) {
      /* Nothing to compute at render time. */
      return args;
    }
    for (int i = 0; i < int(op.args.size()); i++) {
      const Sym &sym = syms[op.args[i]];
      if (sym.base != B_STR || sym.is_array() || !op.reads(i) || sym.str_known) {
        continue;
      }
      if (op.name == "setmessage" && i == 1) {
        continue;
      }
      if (str_fixed.count(op.args[i]) == 0) {
        args.push_back(i);
      }
    }
    return args;
  }

  /* Emit an instruction. An instruction that needs the text of strings which have several
   * possible values is emitted once for each combination of the values. */
  bool emit_op(const Op &op, string &code)
  {
    const vector<int> args = string_content_args(op);
    if (args.empty()) {
      return emit_op_versioned(op, code);
    }
    StrCombos combos;
    if (!str_combinations(op, args, combos)) {
      for (const int arg : args) {
        if (syms[op.args[arg]].str_top) {
          return fail("instruction " + op.name + " uses the string " + syms[op.args[arg]].name +
                      ", which is computed at render time in a way that is not supported on "
                      "this device");
        }
      }
      return fail("instruction " + op.name + " uses strings with too many possible values");
    }
    bool first = true;
    for (const vector<string> &values : combos.values) {
      string condition;
      for (size_t i = 0; i < args.size(); i++) {
        const int sym_index = op.args[args[i]];
        str_fixed[sym_index] = values[i];
        condition += ((i > 0) ? " && " : "") + syms[sym_index].cname + " == " +
                     str_literal(values[i]);
      }
      /* The same symbol can be several arguments. */
      bool consistent = true;
      for (size_t i = 0; i < args.size(); i++) {
        consistent &= (str_fixed[op.args[args[i]]] == values[i]);
      }
      string version;
      const bool ok = !consistent || emit_op_versioned(op, version);
      for (const int arg : args) {
        str_fixed.erase(op.args[arg]);
      }
      if (!ok) {
        return false;
      }
      if (!consistent) {
        continue;
      }
      code += string(first ? "if (" : "else if (") + condition + ") {\n" + version + "}\n";
      first = false;
    }
    return true;
  }

  /* Table of the characters of a string, for getchar(). */
  string str_chars_table(const string &value)
  {
    const string name = "osl_str_chars_" + str_literal(value);
    if (str_tables.insert(name).second) {
      globals_code += "constant int " + name + format("[%d] = {", int(value.size()));
      for (size_t i = 0; i < value.size(); i++) {
        globals_code += format("%d", int((unsigned char)value[i])) +
                        ((i + 1 < value.size()) ? ", " : "");
      }
      globals_code += "};\n";
    }
    return name;
  }

  /* Table of all parts of a string, for substr() with positions computed at render time. */
  string str_substr_table(const string &value)
  {
    const string name = "osl_str_substr_" + str_literal(value);
    if (str_tables.insert(name).second) {
      const int len = int(value.size());
      string table;
      for (int b = 0; b <= len; b++) {
        for (int n = 0; n <= len; n++) {
          table += str_literal(str_substr(value, b, n)) + ", ";
        }
        table += "\n";
      }
      globals_code += "constant int " + name + format("[%d] = {\n", (len + 1) * (len + 1)) +
                      table + "};\n";
    }
    return name;
  }
  std::set<string> str_tables;

  bool emit_string_op(const Op &op, string &code)
  {
    const string &name = op.name;
    const int nargs = int(op.args.size());
    const Sym &dst = syms[op.args[0]];

    if (is_string_function_op(name)) {
      if (dst.str_top || dst.str_known) {
        return true;
      }
      vector<string> values;
      for (int i = 1; i < nargs; i++) {
        if (syms[op.args[i]].base == B_STR) {
          string value;
          if (!string_arg(op, i, value)) {
            return false;
          }
          values.push_back(value);
        }
      }
      string result;
      if (name == "concat") {
        for (const string &value : values) {
          result += value;
        }
      }
      else if (name == "format") {
        if (values.empty() || !format_string(op, 1, values, result)) {
          return fail("instruction format with numbers computed at render time is not "
                      "supported on this device");
        }
      }
      else {
        if (nargs != 4 || values.size() != 1) {
          return fail("malformed instruction substr");
        }
        const Sym &start = syms[op.args[2]];
        const Sym &length = syms[op.args[3]];
        if (start.kind == SYM_CONST && length.kind == SYM_CONST) {
          result = str_substr(values[0], start.ival[0], length.ival[0]);
        }
        else if (values[0].empty()) {
          result = "";
        }
        else {
          string s, l;
          if (!read(op.args[2], {B_INT, false}, s) || !read(op.args[3], {B_INT, false}, l)) {
            return false;
          }
          return write(op.args[0],
                       str_substr_table(values[0]) + "[osl_substr_index(" + s + ", " + l + ", " +
                           format("%d", int(values[0].size())) + ")]",
                       {B_STR, false},
                       code);
        }
      }
      return write(op.args[0], str_literal(result), {B_STR, false}, code);
    }

    if (name == "strlen") {
      string s;
      if (nargs != 2 || !string_arg(op, 1, s)) {
        return false;
      }
      return write(op.args[0], format("%d", int(s.size())), {B_INT, false}, code);
    }
    if (name == "startswith" || name == "endswith") {
      string s, sub;
      if (nargs != 3 || !string_arg(op, 1, s) || !string_arg(op, 2, sub)) {
        return false;
      }
      bool result = s.size() >= sub.size();
      if (result) {
        result = (name == "startswith") ? s.compare(0, sub.size(), sub) == 0 :
                                          s.compare(s.size() - sub.size(), sub.size(), sub) == 0;
      }
      return write(op.args[0], result ? "1" : "0", {B_INT, false}, code);
    }
    if (name == "stoi" || name == "stof") {
      string s;
      if (nargs != 2 || !string_arg(op, 1, s)) {
        return false;
      }
      if (name == "stoi") {
        const long long value = strtoll(s.c_str(), nullptr, 10);
        Sym number;
        number.base = B_INT;
        number.ival.push_back(int(std::min(std::max(value, (long long)INT_MIN), (long long)INT_MAX)));
        return write(op.args[0], literal(number, 0), {B_INT, false}, code);
      }
      return write(op.args[0], float_literal(strtof(s.c_str(), nullptr)), {B_FLOAT, false}, code);
    }
    if (name == "getchar") {
      string s, index;
      if (nargs != 3 || !string_arg(op, 1, s) || !read(op.args[2], {B_INT, false}, index)) {
        return false;
      }
      if (s.empty()) {
        return write(op.args[0], "0", {B_INT, false}, code);
      }
      const string table = str_chars_table(s);
      const string length = format("%d", int(s.size()));
      return write(op.args[0],
                   "((uint(" + index + ") < " + length + "u) ? " + table + "[osl_idx(" + index +
                       ", " + length + ")] : 0)",
                   {B_INT, false},
                   code);
    }
    if (name == "regex_search" || name == "regex_match") {
      /* regex result subject [results] pattern */
      string subject, pattern;
      if (nargs < 3 || nargs > 4 || !string_arg(op, 1, subject) ||
          !string_arg(op, nargs - 1, pattern))
      {
        return false;
      }
      std::smatch match;
      bool found = false;
      try {
        const std::regex regex(pattern);
        found = (name == "regex_match") ? std::regex_match(subject, match, regex) :
                                          std::regex_search(subject, match, regex);
      }
      catch (const std::regex_error &) {
        found = false;
      }
      if (nargs == 4) {
        const Sym &results = syms[op.args[2]];
        if (!results.is_array() || results.base != B_INT || results.kind == SYM_CONST) {
          return fail("malformed instruction " + name);
        }
        for (int r = 0; r < results.arraylen; r++) {
          int value = int(pattern.size());
          if (r / 2 < int(match.size())) {
            value = int((r & 1) ? (match[r / 2].second - subject.begin()) :
                                  (match[r / 2].first - subject.begin()));
          }
          code += expr(results) + format("[%d] = %d;\n", r, value);
        }
      }
      return write(op.args[0], found ? "1" : "0", {B_INT, false}, code);
    }
    if (name == "split") {
      /* split count string results [separator [maxsplit]] */
      string str, sep;
      if (nargs < 3 || nargs > 5 || !string_arg(op, 1, str) || (nargs > 3 && !string_arg(op, 3, sep)))
      {
        return false;
      }
      const Sym &results = syms[op.args[2]];
      if (!results.is_array() || results.base != B_STR || results.kind == SYM_CONST ||
          results.str_top)
      {
        return fail("malformed instruction split");
      }
      const int len = results.arraylen;
      const auto version = [&](const int maxsplit, string &out) {
        const vector<string> pieces = str_split(str, sep, maxsplit, len);
        for (size_t i = 0; i < pieces.size(); i++) {
          out += expr(results) + format("[%d] = ", int(i)) + str_literal(pieces[i]) + ";\n";
        }
        return write(op.args[0], format("%d", int(pieces.size())), {B_INT, false}, out);
      };
      if (nargs < 5) {
        return version(len, code);
      }
      const Sym &maxsplit = syms[op.args[4]];
      if (maxsplit.kind == SYM_CONST && maxsplit.base == B_INT) {
        return version(maxsplit.ival[0], code);
      }
      string count;
      if (!read(op.args[4], {B_INT, false}, count)) {
        return false;
      }
      code += "switch (clamp(" + count + format(", 0, %d)) {\n", len);
      for (int i = 0; i <= len; i++) {
        string out;
        if (!version(i, out)) {
          return false;
        }
        code += format("case %d: {\n", i) + out + "break;\n}\n";
      }
      code += "}\n";
      return true;
    }
    return fail("instruction " + name + " is not supported on this device");
  }

  /* Messages: values that a shader stores by name and reads back. A message keeps the type and
   * the value of the first time it is set, as in the OSL runtime. */
  struct Message {
    int id;
    Base base;
    string type_name;
    int arraylen;
  };
  std::map<string, Message> messages;

  void analyze_messages()
  {
    foreach_live_op([&](const Op &op) {
      if (op.name != "setmessage" || op.args.size() != 2) {
        return;
      }
      const Sym &name = syms[op.args[0]];
      const Sym &value = syms[op.args[1]];
      if (name.base != B_STR || name.is_array()) {
        return;
      }
      for (const string &message : name.str_values) {
        if (messages.count(message) == 0) {
          const int id = int(messages.size());
          messages[message] = {id, value.base, value.type_name, value.arraylen};
        }
      }
    });
  }

  bool emit_message(const Op &op, string &code)
  {
    const int nargs = int(op.args.size());
    if (op.name == "setmessage") {
      string name;
      if (nargs != 2 || !string_arg(op, 0, name)) {
        return false;
      }
      const auto it = messages.find(name);
      const Sym &value = syms[op.args[1]];
      if (it == messages.end() || it->second.base != value.base ||
          it->second.type_name != value.type_name || it->second.arraylen != value.arraylen)
      {
        /* The message exists with another type: it keeps its value. */
        return true;
      }
      const Message &message = it->second;
      const string var = format("msg_%d", message.id);
      code += "if (" + var + "_state == 0) {\n";
      if (message.base != B_CLOSURE && message.base != B_OTHER) {
        if (value.base == B_STR && value.str_top) {
          return fail("the string " + value.name +
                      " is computed at render time in a way that is not supported on this device");
        }
        if (value.is_array()) {
          string element;
          if (!convert(expr(value) + "[i]", vt(value), {value.base, false}, element)) {
            return false;
          }
          code += format("for (int i = 0; i < %d; i++) {\n", value.arraylen) + var + "[i] = " +
                  element + ";\n}\n";
        }
        else {
          string v;
          if (!read(op.args[1], {value.base, false}, v)) {
            return false;
          }
          code += var + " = " + v + ";\n";
        }
      }
      code += var + "_state = 2;\n}\n";
      return true;
    }

    /* getmessage result [source] name value */
    if (nargs != 3 && nargs != 4) {
      return fail("malformed instruction getmessage");
    }
    string name, source;
    if (!string_arg(op, nargs - 2, name) || (nargs == 4 && !string_arg(op, 1, source))) {
      return false;
    }
    const Sym &value = syms[op.args[nargs - 1]];
    const auto it = messages.find(name);
    /* Results of trace() do not exist: there is no scene to trace from a camera shader. */
    if (source == "trace" || it == messages.end()) {
      return write(op.args[0], "0", {B_INT, false}, code);
    }
    const Message &message = it->second;
    const string var = format("msg_%d", message.id);
    if (message.base != value.base || message.type_name != value.type_name ||
        message.arraylen != value.arraylen)
    {
      return write(op.args[0], "0", {B_INT, false}, code);
    }
    string found;
    if (message.base != B_CLOSURE && message.base != B_OTHER &&
        !(value.base == B_STR && (value.str_top || value.str_known)))
    {
      if (value.is_array()) {
        if (value.kind == SYM_CONST || value.alias != -1 || value.promoted) {
          return fail("unsupported assignment to " + value.name);
        }
        string element;
        if (!convert(var + "[i]", {value.base, false}, vt(value), element)) {
          return false;
        }
        found += format("for (int i = 0; i < %d; i++) {\n", value.arraylen) + value.cname +
                 "[i] = " + element + ";\n}\n";
      }
      else if (!write(op.args[nargs - 1], var, {value.base, false}, found)) {
        return false;
      }
    }
    string found_result, missing_result;
    if (!write(op.args[0], "1", {B_INT, false}, found_result) ||
        !write(op.args[0], "0", {B_INT, false}, missing_result))
    {
      return false;
    }
    code += "if (" + var + "_state == 2) {\n" + found + found_result + "}\nelse {\n" +
            missing_result + "}\n";
    return true;
  }

  bool emit_op_versioned(const Op &op, string &code)
  {
    const string &name = op.name;
    const int nargs = int(op.args.size());

    /* No operation. */
    if (name == "nop" || name == "end" || name == "useparam" || name == "printf" ||
        name == "fprintf" || name == "warning" || name == "error" || name == "closure")
    {
      return true;
    }
    if (nargs == 0) {
      return fail("instruction " + name + " is not supported on this device");
    }

    const Sym &dst = syms[op.args[0]];
    if (dst.base == B_CLOSURE && op.writes(0)) {
      /* Closures have no meaning for a camera and cannot be read back. */
      return true;
    }
    if (nargs < 2 && (name == "filterwidth" || name == "area" || name == "calculatenormal" ||
                      name == "arraylength"))
    {
      return fail("malformed instruction " + name);
    }
    if (is_string_function_op(name) || name == "strlen" || name == "startswith" ||
        name == "endswith" || name == "stoi" || name == "stof" || name == "getchar" ||
        name == "regex_search" || name == "regex_match" || name == "split")
    {
      return emit_string_op(op, code);
    }
    if (name == "setmessage" || name == "getmessage") {
      return emit_message(op, code);
    }
    if (name == "trace" || name == "pointcloud_search" || name == "pointcloud_get" ||
        name == "pointcloud_write")
    {
      /* There is no scene to trace from a camera shader, and there are no point clouds. */
      return write(op.args[0], "0", {B_INT, false}, code);
    }

    if (name == "assign" || name == "arraycopy") {
      return emit_assign(op, code);
    }
    if (name == "add" || name == "sub" || name == "mul" || name == "div" || name == "neg") {
      return emit_arithmetic(op, code);
    }
    if (name == "eq" || name == "neq" || name == "lt" || name == "le" || name == "gt" ||
        name == "ge")
    {
      return emit_compare(op, code);
    }
    if (name == "bitand" || name == "bitor" || name == "xor" || name == "compl" ||
        name == "shl" || name == "shr" || name == "and" || name == "or" || name == "not")
    {
      return emit_integer(op, code);
    }
    if (name == "mod") {
      if (dst.base == B_INT) {
        return emit_function(op, "osl_mod", false, code);
      }
      return emit_function(op, "osl_fmod", false, code);
    }

    /* Functions applied to each component. */
    {
      static const std::map<string, std::pair<const char *, bool>> functions = {
          {"sin", {"osl_sin", true}},
          {"cos", {"osl_cos", true}},
          {"tan", {"osl_tan", true}},
          {"asin", {"osl_asin", true}},
          {"acos", {"osl_acos", true}},
          {"atan", {"osl_atan", true}},
          {"atan2", {"osl_atan2", true}},
          {"sinh", {"osl_sinh", true}},
          {"cosh", {"osl_cosh", true}},
          {"tanh", {"osl_tanh", true}},
          {"exp", {"osl_exp", true}},
          {"exp2", {"osl_exp2", true}},
          {"expm1", {"osl_expm1", true}},
          {"log", {"osl_log", true}},
          {"log2", {"osl_log2", true}},
          {"log10", {"osl_log10", true}},
          {"logb", {"osl_logb", false}},
          {"sqrt", {"osl_sqrt", true}},
          {"inversesqrt", {"osl_inversesqrt", true}},
          {"cbrt", {"osl_cbrt", true}},
          {"abs", {"osl_abs", true}},
          {"fabs", {"osl_abs", true}},
          {"erf", {"osl_erf", true}},
          {"erfc", {"osl_erfc", true}},
          {"floor", {"osl_floor", false}},
          {"ceil", {"osl_ceil", false}},
          {"round", {"osl_round", false}},
          {"trunc", {"osl_trunc", false}},
          {"sign", {"osl_sign", false}},
          {"pow", {"osl_pow", true}},
          {"fmod", {"osl_fmod", true}},
          {"min", {"osl_min", true}},
          {"max", {"osl_max", true}},
          {"mix", {"osl_mix", true}},
          {"step", {"osl_step", false}},
          {"smoothstep", {"osl_smoothstep", true}},
          {"clamp", {"osl_clamp", true}},
          {"degrees", {"osl_degrees", true}},
          {"radians", {"osl_radians", true}},
      };
      const auto it = functions.find(name);
      if (it != functions.end()) {
        if (dst.base == B_INT && name != "abs" && name != "fabs" && name != "min" &&
            name != "max" && name != "clamp")
        {
          /* Float function with the result converted on assignment. */
          string args;
          if (!read_args(op, {B_FLOAT, false}, args)) {
            return false;
          }
          return write(op.args[0], string(it->second.first) + "(" + args + ")", {B_FLOAT, false},
                       code);
        }
        if (name == "smoothstep" && dst.base != B_FLOAT) {
          return fail("instruction smoothstep is not supported for " + dst.type_name);
        }
        return emit_function(op, it->second.first, it->second.second, code);
      }
    }

    if (name == "select") {
      if (nargs != 4) {
        return fail("malformed instruction select");
      }
      VT t = {dst.base, false};
      if (t.b != B_INT && t.b != B_FLOAT && t.b != B_VEC) {
        return fail("instruction select is not supported for " + dst.type_name);
      }
      t.d = dst.dual && (syms[op.args[1]].dual || syms[op.args[2]].dual);
      string a, b, c;
      if (!read(op.args[1], t, a) || !read(op.args[2], t, b) ||
          !read(op.args[3], {t.b, false}, c))
      {
        return false;
      }
      return write(op.args[0], "osl_select(" + a + ", " + b + ", " + c + ")", t, code);
    }

    if (name == "isnan" || name == "isinf" || name == "isfinite") {
      return emit_fixed(op, "osl_" + name, "if", false, code);
    }
    if (name == "dot") {
      return emit_fixed(op, "osl_dot", "fvv", true, code);
    }
    if (name == "cross") {
      return emit_fixed(op, "osl_cross", "vvv", true, code);
    }
    if (name == "length") {
      return emit_fixed(op, "osl_length", "fv", true, code);
    }
    if (name == "distance" && nargs == 3) {
      return emit_fixed(op, "osl_distance", "fvv", true, code);
    }
    if (name == "normalize") {
      return emit_fixed(op, "osl_normalize", "vv", true, code);
    }
    if (name == "compref") {
      return emit_fixed(op, "osl_comp", "fvi", true, code);
    }
    if (name == "determinant") {
      return emit_fixed(op, "osl_mat_determinant", "fm", false, code);
    }
    if (name == "transpose") {
      return emit_fixed(op, "osl_mat_transpose", "mm", false, code);
    }
    if (name == "mxcompref") {
      return emit_fixed(op, "osl_mat_comp", "fmii", false, code);
    }

    if (name == "compassign") {
      /* compassign vector index value */
      if (nargs != 3 || dst.base != B_VEC || dst.is_array()) {
        return fail("malformed instruction compassign");
      }
      string index, value;
      if (!read(op.args[1], {B_INT, false}, index) || !read(op.args[2], {B_FLOAT, dst.dual}, value))
      {
        return false;
      }
      code += dst.cname + " = osl_setcomp(" + dst.cname + ", " + index + ", " + value + ");\n";
      return true;
    }
    if (name == "mxcompassign") {
      if (nargs != 4 || dst.base != B_MAT || dst.is_array()) {
        return fail("malformed instruction mxcompassign");
      }
      string row, col, value;
      if (!read(op.args[1], {B_INT, false}, row) || !read(op.args[2], {B_INT, false}, col) ||
          !read(op.args[3], {B_FLOAT, false}, value))
      {
        return false;
      }
      code += dst.cname + " = osl_mat_setcomp(" + dst.cname + ", " + row + ", " + col + ", " +
              value + ");\n";
      return true;
    }

    if (name == "aref") {
      /* aref result array index */
      if (nargs != 3) {
        return fail("malformed instruction aref");
      }
      const Sym &array = syms[op.args[1]];
      string index;
      if (dst.base == B_CLOSURE) {
        return true;
      }
      if (!array.is_array() || array.arraylen <= 0 || array.base == B_CLOSURE ||
          array.base == B_OTHER || (array.base == B_STR && array.str_top) ||
          !read(op.args[2], {B_INT, false}, index))
      {
        return fail("unsupported array " + array.name);
      }
      const Sym &source = (array.alias != -1) ? syms[array.alias] : array;
      return write(op.args[0], element(array, index), vt(source), code);
    }
    if (name == "aassign") {
      /* aassign array index value */
      if (nargs != 3) {
        return fail("malformed instruction aassign");
      }
      string index, value;
      if (dst.promoted) {
        /* The value is part of the constant array. */
        return true;
      }
      if (dst.base == B_CLOSURE || (dst.base == B_STR && dst.str_top)) {
        return true;
      }
      if (!dst.is_array() || dst.arraylen <= 0 || dst.alias != -1 || dst.kind == SYM_CONST ||
          dst.base == B_OTHER)
      {
        return fail("unsupported array " + dst.name);
      }
      if (!read(op.args[1], {B_INT, false}, index) || !read(op.args[2], vt(dst), value)) {
        return false;
      }
      code += element(dst, index) + " = " + value + ";\n";
      return true;
    }
    if (name == "arraylength") {
      return write(op.args[0],
                   format("%d", std::max(syms[op.args[1]].arraylen, 0)),
                   {B_INT, false},
                   code);
    }

    if (name == "sincos") {
      /* sincos angle sine cosine */
      if (nargs != 3) {
        return fail("malformed instruction sincos");
      }
      const Sym &angle = syms[op.args[0]];
      /* The angle may be one of the results, so compute both before assigning. */
      code += "{\n";
      for (int i = 1; i <= 2; i++) {
        const Sym &result = syms[op.args[i]];
        const VT t = {result.base == B_VEC ? B_VEC : B_FLOAT, result.dual && angle.dual};
        string a;
        if (!read(op.args[0], t, a)) {
          return false;
        }
        code += ctype(t) + format(" sincos_%d = ", i) + ((i == 1) ? "osl_sin(" : "osl_cos(") + a +
                ");\n";
      }
      for (int i = 1; i <= 2; i++) {
        const Sym &result = syms[op.args[i]];
        const VT t = {result.base == B_VEC ? B_VEC : B_FLOAT, result.dual && angle.dual};
        if (!write(op.args[i], format("sincos_%d", i), t, code)) {
          return false;
        }
      }
      code += "}\n";
      return true;
    }

    if (name == "Dx" || name == "Dy" || name == "Dz") {
      if (nargs != 2) {
        return fail("malformed instruction " + name);
      }
      const Sym &src = syms[op.args[1]];
      const VT t = {src.base == B_VEC ? B_VEC : B_FLOAT, false};
      string value = zero(t);
      if (src.dual && name != "Dz") {
        value = expr(src) + ((name == "Dx") ? ".dx" : ".dy");
      }
      return write(op.args[0], value, t, code);
    }
    if (name == "filterwidth") {
      const Sym &src = syms[op.args[1]];
      const VT t = {src.base == B_VEC ? B_VEC : B_FLOAT, false};
      return write(op.args[0],
                   src.dual ? "osl_filterwidth(" + expr(src) + ")" : zero(t),
                   t,
                   code);
    }
    if (name == "area") {
      const Sym &src = syms[op.args[1]];
      return write(op.args[0],
                   src.dual ? "length(cross(" + expr(src) + ".dx, " + expr(src) + ".dy))" : "0.0f",
                   {B_FLOAT, false},
                   code);
    }
    if (name == "calculatenormal") {
      const Sym &src = syms[op.args[1]];
      return write(op.args[0],
                   src.dual ? "cross(" + expr(src) + ".dx, " + expr(src) + ".dy)" :
                              "float3(0.0f)",
                   {B_VEC, false},
                   code);
    }

    if (name == "transform" || name == "transformv" || name == "transformn") {
      return emit_transform(op, code);
    }
    if (name == "matrix") {
      return emit_matrix(op, code);
    }
    if (name == "getmatrix") {
      /* getmatrix result from to matrix */
      if (nargs != 4) {
        return fail("malformed instruction getmatrix");
      }
      string from, to;
      bool known;
      if (!string_arg(op, 1, from) || !string_arg(op, 2, to)) {
        return false;
      }
      string value = space_to_space(from, to, known);
      if (value.empty()) {
        value = "osl_mat_diag(1.0f)";
      }
      return write(op.args[3], value, {B_MAT, false}, code) &&
             write(op.args[0], known ? "1" : "0", {B_INT, false}, code);
    }
    if (name == "color" || name == "point" || name == "vector" || name == "normal") {
      return emit_construct(op, code);
    }

    if (name == "luminance") {
      float m[3][3], lum[3];
      color_system(m, lum);
      string c;
      const bool dual = (nargs == 2) && dst.dual && syms[op.args[1]].dual;
      if (nargs != 2 || !read(op.args[1], {B_VEC, dual}, c)) {
        return fail("malformed instruction luminance");
      }
      const string weights = "float3(" + float_literal(lum[0]) + ", " + float_literal(lum[1]) +
                             ", " + float_literal(lum[2]) + ")";
      return write(op.args[0],
                   dual ? "osl_dot(" + c + ", osl_dual(" + weights + "))" :
                          "dot(" + c + ", " + weights + ")",
                   {B_FLOAT, dual},
                   code);
    }
    if (name == "wavelength_color") {
      string lambda;
      if (nargs != 2 || !read(op.args[1], {B_FLOAT, false}, lambda)) {
        return fail("malformed instruction wavelength_color");
      }
      if (options.cie_color_match == nullptr) {
        return fail("instruction wavelength_color is not supported on this device");
      }
      use_wavelength = true;
      return write(op.args[0],
                   "max(" + xyz_to_rgb_expr("osl_wavelength_xyz(" + lambda + ")") +
                       " * (1.0f / 2.52f), float3(0.0f))",
                   {B_VEC, false},
                   code);
    }
    if (name == "blackbody") {
      string t;
      if (nargs != 2 || !read(op.args[1], {B_FLOAT, false}, t)) {
        return fail("malformed instruction blackbody");
      }
      if (options.cie_color_match == nullptr) {
        return fail("instruction blackbody is not supported on this device");
      }
      use_wavelength = true;
      /* Below the Draper point there is no visible emission. */
      return write(op.args[0],
                   "((" + t + " < 800.0f) ? float3(1.0e-6f, 0.0f, 0.0f) : max(" +
                       xyz_to_rgb_expr("osl_blackbody_xyz(" + t + ")") + ", float3(0.0f)))",
                   {B_VEC, false},
                   code);
    }
    if (name == "hash") {
      /* The integer hash of numbers. Strings hash differently and are not supported. */
      const int count = nargs - 1;
      if (count < 1 || count > 2) {
        return fail("malformed instruction hash");
      }
      const Sym &p = syms[op.args[1]];
      if (p.base == B_STR && count == 1) {
        /* The hash of the string itself, which the host computes. */
        string value;
        if (!string_arg(op, 1, value)) {
          return false;
        }
        if (!options.string_hash) {
          return fail("instruction hash is not supported for strings on this device");
        }
        Sym number;
        number.base = B_INT;
        number.ival.push_back(options.string_hash(value));
        return write(op.args[0], literal(number, 0), {B_INT, false}, code);
      }
      if (p.base != B_INT && p.base != B_FLOAT && p.base != B_VEC) {
        return fail("instruction hash is not supported for " + p.type_name);
      }
      string args;
      string function;
      if (p.base == B_INT && count == 1) {
        function = "osl_hash_i1";
        if (!read(op.args[1], {B_INT, false}, args)) {
          return false;
        }
      }
      else {
        const bool is_vector = (p.base == B_VEC);
        function = format("osl_hash_f%d", (is_vector ? 3 : 1) + (count - 1));
        if (!read(op.args[1], {is_vector ? B_VEC : B_FLOAT, false}, args)) {
          return false;
        }
        if (count == 2) {
          string t;
          if (!read(op.args[2], {B_FLOAT, false}, t)) {
            return false;
          }
          args += ", " + t;
        }
      }
      return write(op.args[0], function + "(" + args + ")", {B_INT, false}, code);
    }
    if (name == "transformc") {
      /* transformc result from to color */
      string from, to, c;
      const bool dual = (nargs == 4) && dst.dual && syms[op.args[3]].dual;
      if (nargs != 4 || !string_arg(op, 1, from) || !string_arg(op, 2, to) ||
          !read(op.args[3], {B_VEC, dual}, c))
      {
        return fail("malformed instruction transformc");
      }
      string value;
      if (!color_to_rgb(op, from, c, value) || !color_from_rgb(op, to, value, value)) {
        return false;
      }
      use_color_derivs |= dual;
      return write(op.args[0], value, {B_VEC, dual}, code);
    }

    if (name == "hashnoise" || name == "cellnoise" || name == "noise" || name == "snoise" ||
        name == "pnoise" || name == "psnoise")
    {
      return emit_noise(op, code);
    }
    if (name == "getattribute") {
      return emit_getattribute(op, code);
    }
    if (name == "spline" || name == "splineinverse") {
      return emit_spline(op, code);
    }
    if (name == "dict_find" || name == "dict_next" || name == "dict_value") {
      return emit_dict(op, code);
    }
    if (name == "texture" || name == "texture3d" || name == "environment") {
      return emit_texture(op, code);
    }
    if (name == "gettextureinfo") {
      return emit_gettextureinfo(op, code);
    }

    if (name == "raytype") {
      string type;
      if (nargs != 2 || !string_arg(op, 1, type)) {
        return false;
      }
      return write(op.args[0], (type == "camera") ? "1" : "0", {B_INT, false}, code);
    }
    if (name == "isconnected") {
      /* 2 means connected to something that uses the value, which the outputs that the
       * renderer reads are. */
      const Sym &sym = syms[op.args[nargs - 1]];
      return write(op.args[0],
                   (sym.kind == SYM_OPARAM && is_renderer_output(sym.name)) ? "2" : "0",
                   {B_INT, false},
                   code);
    }
    if (name == "isconstant") {
      const Sym &sym = syms[op.args[nargs - 1]];
      const bool constant = (sym.kind == SYM_CONST) ||
                            (sym.kind == SYM_PARAM && sym.init_begin == -1);
      return write(op.args[0], constant ? "1" : "0", {B_INT, false}, code);
    }
    if (name == "backfacing") {
      return write(op.args[0], "0", {B_INT, false}, code);
    }
    if (name == "surfacearea") {
      return write(op.args[0], "0.0f", {B_FLOAT, false}, code);
    }

    return fail("instruction " + name + " is not supported on this device");
  }

  /* ------------------------------------------------------------------ */
  /* Control flow */

  bool condition(const Op &op, string &result)
  {
    if (op.args.empty()) {
      return fail("malformed instruction " + op.name);
    }
    const Sym &cond = syms[op.args[0]];
    string value;
    if (!read(op.args[0], {cond.base == B_INT ? B_INT : B_FLOAT, false}, value)) {
      return false;
    }
    result = (cond.base == B_INT) ? "(" + value + " != 0)" : "(" + value + " != 0.0f)";
    return true;
  }

  /* The checks between iterations of a loop: step and condition. */
  bool loop_iterate(const int loop_index, const bool is_continue, string &code)
  {
    const Op &loop = ops[loop_index];
    const int cond_begin = loop.jumps[0];
    const int body_begin = loop.jumps[1];
    const int step_begin = loop.jumps[2];
    const int end = loop.jumps[3];
    const string counter = format("loop_%d", loop_index);

    /* Step and condition do not contain loops or function calls of their own that could be
     * affected by the scopes. */
    string cond;
    if (loop.name == "dowhile") {
      if (!emit_range(step_begin, end, code) || !emit_range(cond_begin, body_begin, code) ||
          !condition(loop, cond))
      {
        return false;
      }
      code += "if (!" + cond + " || ++" + counter + " > OSL_LOOP_LIMIT) {\nbreak;\n}\n";
    }
    else if (is_continue) {
      if (!emit_range(step_begin, end, code)) {
        return false;
      }
    }
    else {
      if (!emit_range(cond_begin, body_begin, code) || !condition(loop, cond)) {
        return false;
      }
      code += "if (!" + cond + " || ++" + counter + " > OSL_LOOP_LIMIT) {\nbreak;\n}\n";
    }
    return true;
  }

  bool emit_range(const int begin, const int end, string &code)
  {
    for (int i = begin; i < end;) {
      const Op &op = ops[i];
      const string &name = op.name;

      if (name == "if") {
        if (op.jumps.size() != 2) {
          return fail("malformed instruction if");
        }
        string cond, then_code, else_code;
        if (!condition(op, cond) || !emit_range(i + 1, op.jumps[0], then_code) ||
            !emit_range(op.jumps[0], op.jumps[1], else_code))
        {
          return false;
        }
        code += "if " + cond + " {\n" + then_code + "}\n";
        if (!else_code.empty()) {
          code += "else {\n" + else_code + "}\n";
        }
        i = op.jumps[1];
      }
      else if (name == "for" || name == "while" || name == "dowhile") {
        if (op.jumps.size() != 4) {
          return fail("malformed instruction " + name);
        }
        const int cond_begin = op.jumps[0];
        const int body_begin = op.jumps[1];
        const int step_begin = op.jumps[2];

        /* Initialization. */
        string loop_code;
        if (!emit_range(i + 1, cond_begin, loop_code)) {
          return false;
        }
        loop_code += format("int loop_%d = 0;\n", i);
        loop_code += "for (;;) {\n";

        scopes.push_back({false, num_scopes++, i, {}});
        bool ok = true;
        if (name == "dowhile") {
          ok = emit_range(body_begin, step_begin, loop_code) && loop_iterate(i, false, loop_code);
        }
        else {
          string step_code;
          ok = loop_iterate(i, false, loop_code) &&
               emit_range(body_begin, step_begin, loop_code) &&
               emit_range(step_begin, op.jumps[3], step_code);
          loop_code += step_code;
        }
        const Scope scope = scopes.back();
        scopes.pop_back();
        if (!ok) {
          return false;
        }
        loop_code += "}\n";

        /* Returns from a function inside the loop continue leaving the function. */
        for (const int function : scope.returned) {
          loop_code += format("if (returned_%d) {\nbreak;\n}\n", function);
        }
        code += "{\n" + loop_code + "}\n";
        i = op.jumps[3];
      }
      else if (name == "functioncall" || name == "functioncall_nr") {
        if (op.jumps.size() != 1) {
          return fail("malformed instruction " + name);
        }
        const int id = num_scopes++;
        scopes.push_back({true, id, -1, {}});
        string body;
        const bool ok = emit_range(i + 1, op.jumps[0], body);
        const Scope scope = scopes.back();
        scopes.pop_back();
        if (!ok) {
          return false;
        }
        if (!scope.returned.empty()) {
          code += format("bool returned_%d = false;\n", id);
        }
        code += "do {\n" + body + "} while (false);\n";
        i = op.jumps[0];
      }
      else if (name == "return" || name == "exit") {
        /* Find the function to return from. */
        int function = -1;
        if (name == "return") {
          for (int s = int(scopes.size()) - 1; s >= 0; s--) {
            if (scopes[s].is_function) {
              function = s;
              break;
            }
          }
        }
        if (function == -1) {
          /* End of the shader. */
          code += "return false;\n";
        }
        else if (function == int(scopes.size()) - 1) {
          code += "break;\n";
        }
        else {
          /* Leave the loops inside the function first. */
          const int id = scopes[function].id;
          scopes[function].returned.insert(id);
          for (size_t s = function + 1; s < scopes.size(); s++) {
            scopes[s].returned.insert(id);
          }
          code += format("returned_%d = true;\nbreak;\n", id);
        }
        i++;
      }
      else if (name == "break" || name == "continue") {
        if (scopes.empty() || scopes.back().is_function) {
          return fail("instruction " + name + " outside of a loop");
        }
        if (name == "continue") {
          if (!loop_iterate(scopes.back().loop_op, true, code)) {
            return false;
          }
          code += "continue;\n";
        }
        else {
          code += "break;\n";
        }
        i++;
      }
      else {
        string op_code;
        if (!emit_op(op, op_code)) {
          return false;
        }
        code += op_code;
        i++;
      }
    }
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* Code generation */

  void assign_names()
  {
    std::set<string> names;
    for (Sym &sym : syms) {
      string name = "s_";
      for (const char c : sym.name) {
        const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                           (c >= 'A' && c <= 'Z');
        if (alnum) {
          name += c;
        }
        else if (name.back() != '_') {
          name += '_';
        }
      }
      string unique = name;
      for (int i = 2; names.count(unique); i++) {
        unique = name + format("_%d", i);
      }
      names.insert(unique);
      sym.cname = sym.is_output ? "o->" + unique : unique;
      sym.field_name = unique;
    }
  }

  /* Comma separated initial values of a symbol. */
  string initializer(const Sym &sym)
  {
    string s;
    const int n = std::max(sym.arraylen, 1);
    for (int i = 0; i < n; i++) {
      if (sym.base == B_MAT) {
        /* Not a function call: this initializes a constant. */
        const float *f = &sym.fval[i * 16];
        s += "{{";
        for (int row = 0; row < 4; row++) {
          s += "float4(" + float_literal(f[row * 4]) + ", " + float_literal(f[row * 4 + 1]) +
               ", " + float_literal(f[row * 4 + 2]) + ", " + float_literal(f[row * 4 + 3]) + ")" +
               ((row < 3) ? ", " : "}}");
        }
        s += (i + 1 < n) ? ",\n" : "";
        continue;
      }
      s += literal(sym, i) + ((i + 1 < n) ? ", " : "");
      if (i % 8 == 7) {
        s += "\n";
      }
    }
    return s;
  }

  bool declare(const Sym &sym, string &code)
  {
    if (!sym.used || sym.is_output || sym.alias != -1 || sym.base == B_CLOSURE ||
        sym.base == B_OTHER)
    {
      /* Structures only exist as their fields, and closures are not computed. */
      return true;
    }
    if (sym.base == B_STR && ((sym.str_known && sym.slot == -1) || sym.str_top)) {
      return true;
    }
    const VT t = vt(sym);

    if (sym.kind == SYM_CONST || sym.promoted ||
        (sym.kind == SYM_PARAM && sym.is_array() && sym.num_writes == 0 && sym.init_begin == -1))
    {
      /* Scalar constants are literals. */
      if (sym.is_array()) {
        globals_code += "constant " + ctype(t) + " " + sym.cname + format("[%d] = {\n", sym.arraylen) +
                        initializer(sym) + "};\n";
      }
      return true;
    }

    if (sym.slot != -1) {
      string value;
      if (sym.base == B_INT) {
        value = format("as_type<int>(prm[%d])", sym.slot);
      }
      else if (sym.base == B_FLOAT) {
        value = format("as_type<float>(prm[%d])", sym.slot);
      }
      else if (sym.base == B_MAT) {
        value = "osl_mat(";
        for (int i = 0; i < 16; i++) {
          value += format("as_type<float>(prm[%d])", sym.slot + i) + ((i < 15) ? ", " : ")");
        }
      }
      else {
        value = format("float3(as_type<float>(prm[%d]), as_type<float>(prm[%d]), "
                       "as_type<float>(prm[%d]))",
                       sym.slot,
                       sym.slot + 1,
                       sym.slot + 2);
      }
      code += ctype(t) + " " + sym.cname + " = " + value + ";\n";
      return true;
    }

    if (sym.is_array()) {
      code += ctype(t) + " " + sym.cname + format("[%d];\n", sym.arraylen);
      if (sym.kind == SYM_PARAM) {
        for (int i = 0; i < sym.arraylen; i++) {
          string value;
          if (!convert(literal(sym, i), {sym.base, false}, t, value)) {
            return false;
          }
          code += sym.cname + format("[%d] = ", i) + value + ";\n";
        }
      }
      else {
        code += format("for (int i = 0; i < %d; i++) {\n", sym.arraylen) + sym.cname +
                "[i] = " + zero(t) + ";\n}\n";
      }
      return true;
    }

    string value;
    if (sym.kind == SYM_GLOBAL) {
      if (sym.name == "P") {
        value = sym.dual ? "DualV{float3(in[0], in[1], in[2]), float3(in[3], in[4], in[5]), "
                           "float3(in[6], in[7], in[8])}" :
                           "float3(in[0], in[1], in[2])";
      }
      else if (sym.name == "N") {
        /* The random numbers for sampling the lens. */
        if (!convert("float3(in[9], in[10], 0.0f)", {B_VEC, false}, t, value)) {
          return false;
        }
      }
      else {
        value = zero(t);
      }
    }
    else if (sym.kind == SYM_PARAM) {
      if (!convert(literal(sym, 0), {sym.base, false}, t, value)) {
        return false;
      }
    }
    else {
      value = zero(t);
    }
    code += ctype(t) + " " + sym.cname + " = " + value + ";\n";
    return true;
  }

  /* The color matching functions, and the spectral functions that use them. */
  string wavelength_code() const
  {
    string s = "constant float3 osl_cie_color_match[81] = {\n";
    for (int i = 0; i < 81; i++) {
      const float *c = options.cie_color_match + i * 3;
      s += "float3(" + float_literal(c[0]) + ", " + float_literal(c[1]) + ", " +
           float_literal(c[2]) + "),\n";
    }
    s += "};\n"
         /* Emission of a black body in W/m^2, integrated over the visible spectrum. */
         "inline float3 osl_blackbody_xyz(float t)\n"
         "{\n"
         "float3 xyz = float3(0.0f);\n"
         "for (int i = 0; i < 81; i++) {\n"
         "const float wlm = (380.0f + 5.0f * float(i)) * 1e-9f;\n"
         "const float wlm5 = wlm * wlm * wlm * wlm * wlm;\n"
         "const float me = (3.74183e-16f / wlm5) / (exp(1.4388e-2f / (wlm * t)) - 1.0f);\n"
         "xyz += me * osl_cie_color_match[i];\n"
         "}\n"
         "return xyz * 5.0e-9f;\n"
         "}\n"
         "inline float3 osl_wavelength_xyz(float lambda_nm)\n"
         "{\n"
         "float ii = (lambda_nm - 380.0f) * (1.0f / 5.0f);\n"
         "const int i = int(ii);\n"
         "if (!(ii >= 0.0f) || i < 0 || i >= 80) {\n"
         "return float3(0.0f);\n"
         "}\n"
         "ii -= float(i);\n"
         "return mix(osl_cie_color_match[i], osl_cie_color_match[i + 1], ii);\n"
         "}\n";
    return s;
  }

  /* Code writing one output of three floats, with optional derivatives. */
  string output_code(const char *name, const int offset, const bool derivs) const
  {
    const auto it = sym_index.find(name);
    if (it == sym_index.end()) {
      return "";
    }
    const Sym &sym = syms[it->second];
    if (sym.kind != SYM_OPARAM || sym.base != B_VEC || sym.is_array()) {
      return "";
    }
    const string value = "o." + sym.field_name;
    string s;
    const auto store = [&](const string &v, const int at) {
      s += format("out[%d] = ", at) + v + format(".x;\nout[%d] = ", at + 1) + v +
           format(".y;\nout[%d] = ", at + 2) + v + ".z;\n";
    };
    if (sym.dual) {
      store(value + ".v", offset);
      if (derivs) {
        store(value + ".dx", offset + 3);
        store(value + ".dy", offset + 6);
      }
    }
    else {
      store(value, offset);
    }
    return s;
  }

  bool generate(OSLCameraTranslateResult &result)
  {
    assign_names();

    /* Symbols. */
    string declarations;
    string outputs_struct = "struct OslCameraOutputs {\n";
    string outputs_init;
    for (const Sym &sym : syms) {
      if (!declare(sym, declarations)) {
        return false;
      }
      if (sym.is_output) {
        if (sym.base == B_CLOSURE || sym.base == B_OTHER) {
          continue;
        }
        const VT t = vt(sym);
        const int count = std::max(sym.arraylen, 1);
        outputs_struct += ctype(t) + " " + sym.field_name +
                          (sym.is_array() ? format("[%d]", count) : "") + ";\n";
        for (int i = 0; i < count; i++) {
          string value;
          if (!convert(literal(sym, i), {sym.base, false}, t, value)) {
            return false;
          }
          outputs_init += "o." + sym.field_name + (sym.is_array() ? format("[%d]", i) : "") +
                          " = " + value + ";\n";
        }
      }
    }
    outputs_struct += "int unused;\n};\n";

    for (const auto &it : messages) {
      const Message &message = it.second;
      const string var = format("msg_%d", message.id);
      declarations += "int " + var + "_state = 0;\n";
      if (message.base == B_CLOSURE || message.base == B_OTHER) {
        continue;
      }
      const VT t = {message.base, false};
      if (message.arraylen > 0) {
        declarations += ctype(t) + " " + var + format("[%d];\n", message.arraylen) +
                        format("for (int i = 0; i < %d; i++) {\n", message.arraylen) + var +
                        "[i] = " + zero(t) + ";\n}\n";
      }
      else {
        declarations += ctype(t) + " " + var + " = " + zero(t) + ";\n";
      }
    }

    /* Defaults of parameters computed from other parameters, then the shader itself. */
    string body;
    for (const Sym &sym : syms) {
      if (sym.init_begin != -1 && sym.slot == -1 && !sym.str_overridden() &&
          (sym.kind == SYM_PARAM || sym.kind == SYM_OPARAM))
      {
        scopes.push_back({true, num_scopes++, -1, {}});
        string init;
        const bool ok = emit_range(sym.init_begin, sym.init_end, init);
        const Scope scope = scopes.back();
        scopes.pop_back();
        if (!ok) {
          return false;
        }
        if (!scope.returned.empty()) {
          body += format("bool returned_%d = false;\n", scope.id);
        }
        body += "do {\n" + init + "} while (false);\n";
      }
    }
    if (!emit_range(main_begin, main_end, body)) {
      return false;
    }

    string source = osl_camera_msl_prelude_1;
    source += osl_camera_msl_prelude_2;
    source += osl_camera_msl_prelude_3;
    if (use_perlin) {
      source += osl_camera_msl_perlin;
    }
    if (use_periodic) {
      source += osl_camera_msl_periodic;
    }
    if (use_simplex) {
      source += osl_camera_msl_simplex;
    }
    if (use_spline) {
      source += osl_camera_msl_spline;
    }
    if (use_color_derivs) {
      source += osl_camera_msl_color_derivs;
    }
    if (use_gabor) {
      source += osl_camera_msl_gabor;
    }
    if (use_wavelength) {
      source += wavelength_code();
    }
    source += globals_code;
    source += outputs_struct;
    /* Returns true if it wrote a request for an image lookup instead of finishing. */
    source += "static bool osl_camera_main(constant void *lp, device const uint *prm, "
              "thread const float *in, thread float *out, thread OslCameraOutputs *o)\n{\n";
    if (use_images) {
      source += "int tex_next = 0;\nconst int tex_avail = int(in[13]);\n";
    }
    source += declarations;
    source += body;
    source += "return false;\n}\n\n";
    source += "[[visible]] void cycles_metal_osl_camera(constant void *lp, device const uint *prm, "
              "thread const float *in, thread float *out)\n{\n";
    source += "OslCameraOutputs o;\n";
    source += outputs_init;
    /* The request for an image lookup follows the ray. */
    source += format("for (int i = 0; i < %d; i++) {\nout[i] = 0.0f;\n}\n", use_images ? 35 : 21);
    source += "if (osl_camera_main(lp, prm, in, out, &o)) {\nreturn;\n}\n";
    source += output_code("position", 0, !explicit_derivs);
    source += output_code("direction", 9, !explicit_derivs);
    source += output_code("throughput", 18, false);
    if (explicit_derivs) {
      source += output_code("dPdx", 3, false);
      source += output_code("dPdy", 6, false);
      source += output_code("dDdx", 12, false);
      source += output_code("dDdy", 15, false);
    }
    source += "}\n";

    result.source = source;
    result.slots = slots;
    result.images = images;
    result.num_words = num_words;
    return true;
  }
};

}  // namespace

bool osl_camera_translate_msl(const std::string &bytecode,
                              const OSLCameraTranslateOptions &options,
                              OSLCameraTranslateResult &result,
                              std::string &error)
{
  Translator translator(options);
  if (!translator.translate(bytecode, result)) {
    error = translator.error.empty() ? "unknown error" : translator.error;
    return false;
  }
  return true;
}

CCL_NAMESPACE_END
