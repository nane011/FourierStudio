import numpy as np


class EquationEngine:
    """Tek bir matematiksel denklemi (1D ya da 2D parametrik) güvenli
    şekilde değerlendiren sınıf.

    - "sin(t)"                              -> 1D, skaler değer üretir
    - "sin(t), cos(t)"  /  "[sin(t), cos(t)]" -> 2D parametrik (x, y) üretir

    NoteMapper burada üretilen ham değerleri notaya çevirir, arayüz de
    aynı ham değerleri grafik çizmek için kullanır.
    """

    SAFE_FUNCS = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "pi": np.pi, "e": np.e,
        "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp,
    }

    def __init__(self, equation_str, role="ana ses", color="#39ff14"):
        self.role = role
        self.color = color
        self.set_equation(equation_str)

    def set_equation(self, equation_str):
        self.raw_equation = equation_str
        self.left_expr, self.right_expr = self._split_vector(equation_str)
        self.is_parametric = self.left_expr != self.right_expr

        # PERFORMANS: eval(string, ...) her çağrıda ifadeyi yeniden
        # PARSE EDER ve DERLER — bu, saniyede onlarca kez çağrılan
        # evaluate()/evaluate_array() için gereksiz bir maliyettir.
        # Bunun yerine ifadeyi SADECE denklem değiştiğinde bir kez
        # derleyip (compile) kod nesnesini saklıyoruz; sonraki her
        # eval() çağrısı doğrudan bu hazır bytecode üzerinde çalışır.
        try:
            self._left_code = compile(self.left_expr, "<equation-left>", "eval")
        except Exception as error:
            print(f"Denklem derleme hatası ({self.left_expr}): {error}")
            self._left_code = compile("0.0", "<equation-left>", "eval")

        if self.is_parametric:
            try:
                self._right_code = compile(self.right_expr, "<equation-right>", "eval")
            except Exception as error:
                print(f"Denklem derleme hatası ({self.right_expr}): {error}")
                self._right_code = compile("0.0", "<equation-right>", "eval")
        else:
            self._right_code = self._left_code

    @staticmethod
    def _split_vector(equation_str):
        """[X, Y] ya da X, Y formatındaki 2D denklemi parantez derinliğini
        takip ederek güvenle Sol/Sağ bileşene ayırır. Tek boyutlu denklemde
        her iki tarafa da aynı ifade döner."""

        s = equation_str.strip().replace("^", "**")

        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]

        depth = 0
        split_idx = -1
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                split_idx = i
                break

        if split_idx != -1:
            return s[:split_idx].strip(), s[split_idx + 1:].strip()

        return s, s

    def _eval_code(self, code, t):
        safe_dict = dict(self.SAFE_FUNCS)
        safe_dict["t"] = t
        try:
            return eval(code, {"__builtins__": None}, safe_dict)
        except Exception as error:
            print(f"Denklem çalıştırma hatası: {error}")
            return 0.0

    def evaluate(self, t):
        """t: skaler zaman (saniye). Denklem parametrikse (x, y) tuple,
        değilse tek bir float döner. NoteMapper'ın beklediği format budur."""

        x = self._eval_code(self._left_code, t)
        x = float(x) if np.isscalar(x) else float(np.asarray(x).flat[0])

        if self.is_parametric:
            y = self._eval_code(self._right_code, t)
            y = float(y) if np.isscalar(y) else float(np.asarray(y).flat[0])
            return (x, y)

        return x

    def evaluate_array(self, t_array):
        """Grafik çizimi için: bir zaman dizisi üzerinde vektörel
        (numpy array) değerlendirme. (x_array, y_array_veya_None) döner."""

        x = self._eval_code(self._left_code, t_array)
        x = np.asarray(x, dtype=np.float64)
        if x.shape != t_array.shape:
            fill = float(x.flat[0]) if x.size else 0.0
            x = np.full_like(t_array, fill)

        if self.is_parametric:
            y = self._eval_code(self._right_code, t_array)
            y = np.asarray(y, dtype=np.float64)
            if y.shape != t_array.shape:
                fill = float(y.flat[0]) if y.size else 0.0
                y = np.full_like(t_array, fill)
            return x, y

        return x, None
