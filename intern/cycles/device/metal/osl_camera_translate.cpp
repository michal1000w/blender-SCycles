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
 * - Strings only exist at translation time.
 *
 * Unsupported instructions (closures, textures, most noise types, strings computed at render
 * time) make the translation fail with a message naming the instruction. */

#include "device/metal/osl_camera_translate.h"
#include "device/metal/osl_camera_prelude.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
enum Base { B_INT, B_FLOAT, B_VEC, B_MAT, B_STR, B_OTHER };

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

  bool str_known = false;
  /* Value assigned by the camera, the default is not computed. */
  bool str_is_overridden = false;
  string str;

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

      vector<Token> tokens;
      if (!tokenize(line, tokens)) {
        return fail("malformed bytecode line: " + line);
      }
      if (tokens.empty() || (!tokens[0].quoted && tokens[0].text[0] == '#')) {
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
    if (type == "closure" || type == "struct") {
      /* Followed by the closure or structure type. Structure fields are separate symbols. */
      t++;
      type = "unsupported";
    }
    parse_type(type, sym);
    if (t >= tokens.size()) {
      return fail("malformed symbol");
    }
    sym.name = tokens[t++].text;

    for (; t < tokens.size() && !tokens[t].hint; t++) {
      const Token &token = tokens[t];
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
        "hashnoise",   "cellnoise",  "pnoise",      "psnoise",
        "eq",          "neq",        "lt",          "le",        "gt",         "ge",
        "getattribute", "raytype",   "arraylength", "Dx",        "Dy",         "Dz",
        "filterwidth", "area",       "calculatenormal", "isnan", "isinf",      "isfinite",
        "bitand",      "bitor",      "xor",         "compl",     "shl",        "shr",
        "and",         "or",         "not",         "logb",      "color",      "transformc",
        "wavelength_color", "blackbody", "luminance", "matrix",  "getmatrix",  "determinant",
        "hash",
        "transpose",   "mxcompref",  "mxcompassign", "strlen",   "startswith", "endswith",
        "isconnected", "isconstant", "backfacing",  "surfacearea", "mod",
    };
    return names.count(name) != 0;
  }

  static bool is_deriv_query_op(const string &name)
  {
    return name == "Dx" || name == "Dy" || name == "Dz" || name == "filterwidth" ||
           name == "area" || name == "calculatenormal";
  }

  bool analyze()
  {
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
                (sym.base == B_FLOAT || sym.base == B_VEC) && param.size == sym.num_components()))
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

    /* Strings are constants of the translation. */
    for (Sym &sym : syms) {
      if (sym.base == B_STR && !sym.is_array() && (sym.kind == SYM_CONST || sym.kind == SYM_PARAM))
      {
        sym.str_known = (sym.kind == SYM_CONST || sym.init_begin == -1 || sym.str_is_overridden);
        sym.str = sym.sval[0];
      }
    }
    foreach_live_op([&](const Op &op) {
      if (op.args.empty()) {
        return;
      }
      Sym &dst = syms[op.args[0]];
      if (dst.base != B_STR || dst.is_array() || dst.kind == SYM_CONST || dst.num_writes != 1) {
        return;
      }
      if (op.name == "assign" || op.name == "concat") {
        string value;
        for (size_t i = 1; i < op.args.size(); i++) {
          const Sym &src = syms[op.args[i]];
          if (src.base != B_STR || !src.str_known) {
            return;
          }
          value += src.str;
        }
        dst.str_known = true;
        dst.str = value;
      }
    });

    analyze_derivatives();
    return true;
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

  static bool carries_derivs(const Sym &sym)
  {
    return sym.base == B_FLOAT || sym.base == B_VEC;
  }

  void analyze_derivatives()
  {
    bool explicit_derivs = false;
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

  string literal(const Sym &sym, const int element) const
  {
    switch (sym.base) {
      case B_INT:
        return format("%d", sym.ival[element]);
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
  string expr(const Sym &sym) const
  {
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
    if (sym.base == B_STR || sym.base == B_OTHER) {
      return fail("unsupported use of " + sym.type_name + " " + sym.name);
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
    if (sym.base == B_STR) {
      /* Strings are tracked by the analysis. */
      return true;
    }
    if (sym.base == B_OTHER) {
      return fail("unsupported use of " + sym.type_name + " " + sym.name);
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
    const Sym &sym = syms[op.args[arg]];
    if (sym.base != B_STR || !sym.str_known) {
      return fail("instruction " + op.name + " needs a string that is constant, but " + sym.name +
                  " is computed by the shader");
    }
    value = sym.str;
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
      string sa, sb;
      if (!string_arg(op, 1, sa) || !string_arg(op, 2, sb)) {
        return false;
      }
      if (name != "eq" && name != "neq") {
        return fail("instruction " + name + " is not supported for strings");
      }
      value = ((sa == sb) == (name == "eq")) ? "1" : "0";
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
  string element(const Sym &array, const string &index) const
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
    if (dst.base == B_STR) {
      /* Strings are tracked by the analysis. */
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

  bool emit_noise(const Op &op, string &code)
  {
    /* The name of the function: noise and snoise are unsigned and signed Perlin noise. */
    string name = op.name;
    int first = 1;
    if ((name == "noise" || name == "snoise") && op.args.size() > 1 &&
        syms[op.args[1]].base == B_STR)
    {
      string type;
      if (!string_arg(op, 1, type)) {
        return false;
      }
      first = 2;
      if (type == "hash") {
        name = "hashnoise";
      }
      else if (type == "cell") {
        name = "cellnoise";
      }
      else if (type == "perlin" || type == "snoise") {
        name = "snoise";
      }
      else if (type == "uperlin" || type == "noise") {
        name = "noise";
      }
      else {
        return fail("noise type \"" + type + "\" is not supported on this device");
      }
    }
    if (name != "hashnoise" && name != "cellnoise" && name != "noise" && name != "snoise") {
      return fail("instruction " + op.name + " is not supported on this device");
    }
    const bool perlin = (name == "noise" || name == "snoise");

    const Sym &dst = syms[op.args[0]];
    const int nargs = int(op.args.size()) - first;
    if (nargs < 1 || nargs > 2 || (dst.base != B_FLOAT && dst.base != B_VEC)) {
      return fail("malformed instruction " + op.name);
    }
    const Sym &p = syms[op.args[first]];
    const int dimensions = ((p.base == B_VEC) ? 3 : 1) + (nargs - 1);
    /* Hash and cell noise are not continuous. */
    const bool dual = perlin && dst.dual && any_dual(op, first);
    string args;
    if (!read(op.args[first], {p.base == B_VEC ? B_VEC : B_FLOAT, dual}, args)) {
      return false;
    }
    if (nargs == 2) {
      string t;
      if (!read(op.args[first + 1], {B_FLOAT, dual}, t)) {
        return false;
      }
      args += ", " + t;
    }
    const string function = "osl_" + name + ((dst.base == B_VEC) ? "_v" : "_f") +
                            format("%d", dimensions);
    return write(op.args[0], function + "(" + args + ")", {dst.base, dual}, code);
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
        value = "osl_mat_mul(" + value + ", " + to_common + ")";
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

    VT t = {B_FLOAT, dst.dual && any_dual(op, first) && op.name != "color"};
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

  bool emit_op(const Op &op, string &code)
  {
    const string &name = op.name;
    const int nargs = int(op.args.size());

    /* No operation. */
    if (name == "nop" || name == "end" || name == "useparam" || name == "printf" ||
        name == "fprintf" || name == "warning" || name == "error" || name == "setmessage")
    {
      return true;
    }
    if (nargs == 0) {
      return fail("instruction " + name + " is not supported on this device");
    }

    const Sym &dst = syms[op.args[0]];

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
      };
      const auto it = functions.find(name);
      if (it != functions.end()) {
        if (dst.base == B_INT && name != "abs" && name != "min" && name != "max") {
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
      if (!array.is_array() || array.arraylen <= 0 || array.base == B_STR ||
          array.base == B_OTHER || !read(op.args[2], {B_INT, false}, index))
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
      if (!dst.is_array() || dst.arraylen <= 0 || dst.alias != -1 || dst.kind == SYM_CONST ||
          dst.base == B_STR || dst.base == B_OTHER)
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
      if (nargs != 2 || !read(op.args[1], {B_VEC, false}, c)) {
        return fail("malformed instruction luminance");
      }
      return write(op.args[0],
                   "dot(" + c + ", float3(" + float_literal(lum[0]) + ", " +
                       float_literal(lum[1]) + ", " + float_literal(lum[2]) + "))",
                   {B_FLOAT, false},
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
      if (p.base == B_STR || p.base == B_MAT || p.base == B_OTHER) {
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
      if (nargs != 4 || !string_arg(op, 1, from) || !string_arg(op, 2, to) ||
          !read(op.args[3], {B_VEC, false}, c))
      {
        return fail("malformed instruction transformc");
      }
      string value;
      if (!color_to_rgb(op, from, c, value) || !color_from_rgb(op, to, value, value)) {
        return false;
      }
      return write(op.args[0], value, {B_VEC, false}, code);
    }

    if (name == "hashnoise" || name == "cellnoise" || name == "noise" || name == "snoise" ||
        name == "pnoise" || name == "psnoise")
    {
      return emit_noise(op, code);
    }
    if (name == "getattribute") {
      return emit_getattribute(op, code);
    }

    if (name == "raytype") {
      string type;
      if (nargs != 2 || !string_arg(op, 1, type)) {
        return false;
      }
      return write(op.args[0], (type == "camera") ? "1" : "0", {B_INT, false}, code);
    }
    if (name == "isconnected" || name == "isconstant" || name == "backfacing" ||
        name == "getmessage")
    {
      return write(op.args[0], "0", {B_INT, false}, code);
    }
    if (name == "surfacearea") {
      return write(op.args[0], "0.0f", {B_FLOAT, false}, code);
    }

    /* Strings. */
    if (name == "concat" || name == "format") {
      return true;
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
          code += "return;\n";
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
  string initializer(const Sym &sym) const
  {
    string s;
    const int n = std::max(sym.arraylen, 1);
    for (int i = 0; i < n; i++) {
      s += literal(sym, i) + ((i + 1 < n) ? ", " : "");
      if (i % 8 == 7) {
        s += "\n";
      }
    }
    return s;
  }

  bool declare(const Sym &sym, string &code)
  {
    if (!sym.used || sym.is_output || sym.alias != -1 || sym.base == B_STR) {
      return true;
    }
    if (sym.base == B_OTHER) {
      return fail("shader uses unsupported type " + sym.type_name + " for " + sym.name);
    }
    if (sym.arraylen < 0) {
      return fail("array " + sym.name + " has no fixed length");
    }
    const VT t = vt(sym);

    if (sym.kind == SYM_CONST || sym.promoted ||
        (sym.kind == SYM_PARAM && sym.is_array() && sym.num_writes == 0 && sym.init_begin == -1))
    {
      /* Scalar constants are literals. */
      if (sym.is_array()) {
        if (sym.base == B_MAT) {
          return fail("constant matrix array " + sym.name + " is not supported");
        }
        globals_code += "constant " + ctype(t) + " " + sym.cname +
                        format("[%d] = {\n", sym.arraylen) + initializer(sym) + "};\n";
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
        if (sym.is_array() || (sym.base != B_INT && sym.base != B_FLOAT && sym.base != B_VEC &&
                               sym.base != B_MAT))
        {
          if (sym.base == B_STR) {
            continue;
          }
          return fail("output " + sym.name + " has an unsupported type");
        }
        const VT t = vt(sym);
        string value;
        if (!convert(literal(sym, 0), {sym.base, false}, t, value)) {
          return false;
        }
        outputs_struct += ctype(t) + " " + sym.field_name + ";\n";
        outputs_init += "o." + sym.field_name + " = " + value + ";\n";
      }
    }
    outputs_struct += "int unused;\n};\n";

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

    bool explicit_derivs = false;
    for (const char *name : {"dPdx", "dPdy", "dDdx", "dDdy"}) {
      const auto it = sym_index.find(name);
      explicit_derivs |= (it != sym_index.end() && syms[it->second].kind == SYM_OPARAM);
    }

    string source = osl_camera_msl_prelude_1;
    source += osl_camera_msl_prelude_2;
    source += osl_camera_msl_prelude_3;
    if (use_wavelength) {
      source += wavelength_code();
    }
    source += globals_code;
    source += outputs_struct;
    source += "static void osl_camera_main(constant void *lp, device const uint *prm, "
              "thread const float *in, thread OslCameraOutputs *o)\n{\n";
    source += declarations;
    source += body;
    source += "}\n\n";
    source += "[[visible]] void cycles_metal_osl_camera(constant void *lp, device const uint *prm, "
              "thread const float *in, thread float *out)\n{\n";
    source += "OslCameraOutputs o;\n";
    source += outputs_init;
    source += "osl_camera_main(lp, prm, in, &o);\n";
    source += "for (int i = 0; i < 21; i++) {\nout[i] = 0.0f;\n}\n";
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
