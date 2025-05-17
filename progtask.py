import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import sympy as sp

def incremental_search(f, x_start, x_end, dx):
    x_l = x_start
    x_u = x_l + dx
    iteration = 1
    table = []
    roots = []
    while x_u <= x_end:
        f_xl = f(x_l)
        f_xu = f(x_u)
        product = f_xl * f_xu
        remark = "Go to next interval" if product > 0 else "Root in this interval"
        table.append([iteration, x_l, dx, x_u, f_xl, f_xu, product, remark])
        if product < 0:
            roots.append((x_l + x_u) / 2)
            break
        x_l = x_u
        x_u += dx
        iteration += 1
    return table, roots

def bisection_method(f, x_l, x_u, tol=1e-6, max_iter=100):
    table = []
    iteration = 1
    x_r_old = None
    while iteration <= max_iter:
        x_r = (x_l + x_u) / 2
        f_xl = f(x_l)
        f_xr = f(x_r)
        f_xu = f(x_u)
        if x_r_old is not None:
            error = abs((x_r - x_r_old) / x_r) * 100  # as percent
        else:
            error = None  # or set to a large value or '-'
        product = f_xl * f_xr
        if product > 0:
            product_str = "> 0"
        elif product < 0:
            product_str = "< 0"
        else:
            product_str = "= 0"
        remark = "2nd subinterval" if product > 0 else "1st subinterval"
        table.append([
            iteration, x_l, x_r, x_u, f_xl, f_xr,
            error if error is not None else "-", product_str, remark
        ])
        if error is not None and error <= tol:
            return table, x_r
        if f_xl * f_xr < 0:
            x_u = x_r
        else:
            x_l = x_r
        x_r_old = x_r
        iteration += 1
    return table, x_r

def regula_falsi_method(f, x_l, x_u, tol=1e-6, max_iter=100):
    table = []
    iteration = 1
    x_r_old = None
    while iteration <= max_iter:
        f_xl = f(x_l)
        f_xu = f(x_u)
        x_r = (x_u * f_xl - x_l * f_xu) / (f_xl - f_xu)
        f_xr = f(x_r)
        if x_r_old is not None:
            error = abs((x_r - x_r_old) / x_r)
        else:
            error = None
        product = f_xl * f_xr
        if product > 0:
            product_str = '>0'
            remark = 'xR = xL'
        elif product < 0:
            product_str = '<0'
            remark = 'xR = xU'
        else:
            product_str = '=0'
            remark = 'Root found'
        table.append([
            iteration, x_l, x_u, x_r, error if error is not None else '-', f_xl, f_xu, f_xr, product_str, remark
        ])
        if f_xr == 0 or (error is not None and error <= tol):
            return table, x_r
        if product < 0:
            x_u = x_r
        elif product > 0:
            x_l = x_r
        else:
            return table, x_r
        x_r_old = x_r
        iteration += 1
    return table, x_r

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    table = []
    iteration = 1
    ea = None
    while iteration <= max_iter:
        f_x0 = f(x0)
        f_x1 = f(x1)
        if f_x1 - f_x0 == 0:
            break  # Avoid division by zero
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        f_x2 = f(x2)
        if iteration > 1:
            ea = abs((x2 - x1) / x2)
        else:
            ea = None
        table.append([
            iteration, x0, x1, x2, ea if ea is not None else '-', f_x0, f_x1, f_x2
        ])
        if ea is not None and ea <= tol:
            return table, x2
        x0, x1 = x1, x2
        iteration += 1
    return table, x2

def newton_raphson_method(f, df, x0, tol=1e-6, max_iter=100):
    table = []
    xi = x0
    fxi = f(xi)
    dfxi = df(xi)
    table.append([0, xi, '-', fxi, dfxi])  # First row, no error
    for iteration in range(1, max_iter+1):
        if dfxi == 0:
            break  # Avoid division by zero
        xi_next = xi - fxi / dfxi
        fxi_next = f(xi_next)
        dfxi_next = df(xi_next)
        ea = abs((xi_next - xi) / xi_next)  # error as fraction
        table.append([
            iteration, xi_next, ea, fxi_next, dfxi_next
        ])
        if ea <= tol:
            return table, xi_next
        xi = xi_next
        fxi = fxi_next
        dfxi = dfxi_next
    return table, xi

def derive_function():
    func_str = func_entry.get()
    x = sp.symbols('x')
    try:
        # Parse the function string using sympy
        func_sympy = sp.sympify(func_str, locals={'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'log': sp.log, 'exp': sp.exp, 'sqrt': sp.sqrt, 'abs': sp.Abs, 'pi': sp.pi, 'e': sp.E})
        deriv_sympy = sp.diff(func_sympy, x)
        # Convert sympy expression to a string suitable for eval with numpy
        deriv_str = str(deriv_sympy)
        # Replace sympy functions with numpy equivalents for eval
        deriv_str = deriv_str.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
        deriv_str = deriv_str.replace('log', 'np.log').replace('exp', 'np.exp').replace('Abs', 'np.abs')
        deriv_str = deriv_str.replace('sqrt', 'np.sqrt')
        deriv_str = deriv_str.replace('E', 'np.e').replace('pi', 'np.pi')
        deriv_str = deriv_str.replace('sec', '1/np.cos')
        deriv_str = deriv_str.replace('csc', '1/np.sin')
        deriv_str = deriv_str.replace('cot', '1/np.tan')
        deriv_entry.delete(0, 'end')
        deriv_entry.insert(0, deriv_str)
    except Exception as e:
        messagebox.showerror("Error", f"Could not derive function: {e}")

def plot_and_solve():
    global last_table, last_table_columns
    try:
        func_str = func_entry.get()
        x_start = float(start_entry.get())
        x_end = float(end_entry.get())
        method = method_var.get()
        allowed_funcs = {
            "np": np,
            "pi": np.pi,
            "e": np.e,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "log": np.log,
            "log10": np.log10,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
        }
        f = lambda x: eval(func_str, allowed_funcs, {"x": x})

        # Clear previous plot and table
        ax.clear()
        for item in results_table.get_children():
            results_table.delete(item)

        # Choose a wider range for context
        x_min = min(-1, x_start)
        x_max = max(3, x_end)
        x = np.linspace(x_min, x_max, 400)
        y = f(x)
        ax.plot(x, y, label=f'f(x) = {func_str}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Numerical Methods')
        ax.grid(True)
        ax.legend()

        if method == "Graphical":
            columns = ["x", "f(x)"]
            last_table = []
            x_table = np.linspace(x_start, x_end, 10)
            y_table = f(x_table)
            for xi, yi in zip(x_table, y_table):
                results_table.insert("", "end", values=(f"{xi:.3f}", f"{yi:.6f}"))
                last_table.append([xi, yi])
            last_table_columns = columns
        elif method == "Incremental":
            dx = float(step_entry.get())
            table, roots = incremental_search(f, x_start, x_end, dx)
            columns = ["Iter", "x_l", "Δx", "x_u", "f(x_l)", "f(x_u)", "f(x_l)*f(x_u)", "Remark"]
            last_table = table
            last_table_columns = columns
            for row in table:
                results_table.insert("", "end", values=(
                    row[0], safe_float_fmt(row[1]), safe_float_fmt(row[2]), safe_float_fmt(row[3]),
                    safe_float_fmt(row[4]), safe_float_fmt(row[5]), safe_float_fmt(row[6]), row[7]
                ))
            for root in roots:
                ax.axvline(root, linestyle='dashed', color='red', label=f'Root: {root:.3f}')
        elif method == "Bisection":
            tol = float(tol_entry.get())  # User enters 0.5 for 0.5%
            table, root = bisection_method(f, x_start, x_end, tol)
            columns = ["Iter", "x_l", "x_r", "x_u", "f(x_l)", "f(x_r)", "|e_a| %", "f(x_l)*f(x_r)", "Remark"]
            last_table = table
            last_table_columns = columns
            for row in table:
                results_table.insert("", "end", values=(
                    row[0], safe_float_fmt(row[1]), safe_float_fmt(row[2]), safe_float_fmt(row[3]),
                    safe_float_fmt(row[4]), safe_float_fmt(row[5]),
                    (safe_float_fmt(row[6], ".5f") + " %" if row[6] != '-' else '-'),
                    safe_float_fmt(row[7]), row[8]
                ))
            ax.axvline(root, linestyle='dashed', color='red', label=f'Root: {root:.3f}')
        elif method == "Regula-Falsi":
            tol = float(tol_entry.get())
            table, root = regula_falsi_method(f, x_start, x_end, tol)
            columns = ["Iter", "x_l", "x_u", "x_r", "|e_a| %", "f(x_l)", "f(x_u)", "f(x_r)", "f(x_l)*f(x_r)", "Remark"]
            last_table = table
            last_table_columns = columns
            for row in table:
                results_table.insert("", "end", values=(
                    row[0], safe_float_fmt(row[1]), safe_float_fmt(row[2]), safe_float_fmt(row[3]),
                    safe_float_fmt(row[4]), safe_float_fmt(row[5]), safe_float_fmt(row[6]), safe_float_fmt(row[7]), safe_float_fmt(row[8]), row[9]
                ))
            ax.axvline(root, linestyle='dashed', color='red', label=f'Root: {root:.3f}')
        elif method == "Secant":
            tol = float(tol_entry.get())
            table, root = secant_method(f, x_start, x_end, tol)
            columns = ["Iter", "x_{i-1}", "x_i", "x_{i+1}", "|e_a| %", "f(x_{i-1})", "f(x_i)", "f(x_{i+1})"]
            last_table = table
            last_table_columns = columns
            for row in table:
                results_table.insert("", "end", values=(
                    row[0], safe_float_fmt(row[1]), safe_float_fmt(row[2]), safe_float_fmt(row[3]),
                    safe_float_fmt(row[4]), safe_float_fmt(row[5]), safe_float_fmt(row[6]), safe_float_fmt(row[7])
                ))
            ax.axvline(root, linestyle='dashed', color='red', label=f'Root: {root:.3f}')
        elif method == "Newton-Raphson":
            deriv_str = deriv_entry.get()
            df = lambda x: eval(deriv_str, allowed_funcs, {"x": x})
            x0 = float(init_guess_entry.get())
            tol = float(tol_entry.get())
            table, root = newton_raphson_method(f, df, x0, tol)
            columns = ["Iter", "x_i", "|e_a|", "f(x_i)", "f'(x_i)"]
            last_table = table
            last_table_columns = columns
            for row in table:
                results_table.insert("", "end", values=(
                    row[0],
                    safe_float_fmt(row[1]),
                    '-' if row[2] == '-' else '{:.6f}'.format(row[2]),
                    safe_float_fmt(row[3]),
                    safe_float_fmt(row[4])
                ))
            ax.axvline(root, linestyle='dashed', color='red', label=f'Root: {root:.3f}')
        ax.legend()
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

def update_fields(*args):
    method = method_var.get()
    if method == "Incremental":
        step_label.grid(row=4, column=0, sticky="w")
        step_entry.grid(row=4, column=1, sticky="w")
        tol_label.grid_remove()
        tol_entry.grid_remove()
        deriv_label.grid_remove()
        deriv_entry.grid_remove()
        init_guess_entry.grid_remove()
    elif method == "Bisection" or method == "Regula-Falsi" or method == "Secant":
        tol_label.grid(row=6, column=0, sticky="w")
        tol_entry.grid(row=6, column=1, sticky="w")
        step_label.grid_remove()
        step_entry.grid_remove()
        deriv_label.grid_remove()
        deriv_entry.grid_remove()
        init_guess_entry.grid_remove()
    elif method == "Newton-Raphson":
        deriv_label.grid(row=4, column=0, sticky="w")
        deriv_entry.grid(row=4, column=1, sticky="w")
        tol_label.grid(row=6, column=0, sticky="w")
        tol_entry.grid(row=6, column=1, sticky="w")
        init_guess_entry.grid(row=3, column=1, sticky="w")
        step_label.grid_remove()
        step_entry.grid_remove()
    else:  # Graphical
        step_label.grid_remove()
        step_entry.grid_remove()
        tol_label.grid_remove()
        tol_entry.grid_remove()
        deriv_label.grid_remove()
        deriv_entry.grid_remove()
        init_guess_entry.grid_remove()

def save_table_to_excel():
    if not last_table or not last_table_columns:
        messagebox.showinfo("No Data", "No table data to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df = pd.DataFrame(last_table, columns=last_table_columns)
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Success", f"Table saved to {file_path}")

def save_plot_to_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
    if file_path:
        fig.savefig(file_path)
        messagebox.showinfo("Success", f"Plot image saved to {file_path}")

def safe_float_fmt(val, fmt=".6f"):
    try:
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return str(val)

def zoom_factory(ax, base_scale=1.1):
    def zoom(event):
        # Only zoom if mouse is over the axes
        if event.inaxes != ax:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_xlim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_xlim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', zoom)
    return zoom

def pan_factory(ax):
    press = {'x': None, 'y': None, 'xlim': None, 'ylim': None}

    def on_press(event):
        if event.inaxes != ax:
            return
        press['x'] = event.x
        press['y'] = event.y
        press['xlim'] = ax.get_xlim()
        press['ylim'] = ax.get_ylim()

    def on_release(event):
        press['x'] = None
        press['y'] = None
        press['xlim'] = None
        press['ylim'] = None

    def on_motion(event):
        if press['x'] is None or press['y'] is None:
            return
        if event.inaxes != ax:
            return
        dx = event.x - press['x']
        dy = event.y - press['y']
        scale_x = (press['xlim'][1] - press['xlim'][0]) / ax.bbox.width
        scale_y = (press['ylim'][1] - press['ylim'][0]) / ax.bbox.height
        ax.set_xlim(press['xlim'][0] - dx * scale_x, press['xlim'][1] - dx * scale_x)
        ax.set_ylim(press['ylim'][0] + dy * scale_y, press['ylim'][1] + dy * scale_y)
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

# --- GUI Setup ---
root = tk.Tk()
root.title("Numerical Methods - Unified Window")
root.geometry("1400x700")  # Start with a larger window

frame = ttk.Frame(root, padding=10)
frame.pack(fill=tk.BOTH, expand=True)

# Function input
ttk.Label(frame, text="Function f(x):").grid(row=0, column=0, sticky="w")
func_entry = ttk.Entry(frame, width=30)
func_entry.insert(0, "np.exp(-x)-x")
func_entry.grid(row=0, column=1, padx=5, pady=5)

deriv_label = ttk.Label(frame, text="Derivative f'(x):")
deriv_entry = ttk.Entry(frame, width=30)

ttk.Label(frame, text="Initial x₀:").grid(row=3, column=0, sticky="w")
init_guess_entry = ttk.Entry(frame, width=10)
init_guess_entry.insert(0, "0")
init_guess_entry.grid(row=3, column=1, sticky="w")

# Interval input
ttk.Label(frame, text="Start x:").grid(row=1, column=0, sticky="w")
start_entry = ttk.Entry(frame, width=10)
start_entry.insert(0, "0")
start_entry.grid(row=1, column=1, sticky="w")

ttk.Label(frame, text="End x:").grid(row=2, column=0, sticky="w")
end_entry = ttk.Entry(frame, width=10)
end_entry.insert(0, "1")
end_entry.grid(row=2, column=1, sticky="w")

# Step size (for Incremental)
step_label = ttk.Label(frame, text="Step size:")
step_entry = ttk.Entry(frame, width=10)
step_entry.insert(0, "0.1")

# Tolerance (for Bisection)
tol_label = ttk.Label(frame, text="Tolerance:")
tol_entry = ttk.Entry(frame, width=10)
tol_entry.insert(0, "1e-6")

# Method selection
ttk.Label(frame, text="Method:").grid(row=5, column=0, sticky="w")
method_var = tk.StringVar(value="Graphical")
method_menu = ttk.Combobox(frame, textvariable=method_var, values=["Graphical", "Incremental", "Bisection", "Regula-Falsi", "Secant", "Newton-Raphson"], state="readonly")
method_menu.grid(row=5, column=1, sticky="w")
method_var.trace_add("write", update_fields)

plot_btn = ttk.Button(frame, text="Plot/Solve", command=plot_and_solve)
plot_btn.grid(row=7, column=0, columnspan=2, pady=5)

save_table_btn = ttk.Button(frame, text="Save Table to Excel", command=save_table_to_excel)
save_table_btn.grid(row=8, column=0, columnspan=2, pady=2)

save_plot_btn = ttk.Button(frame, text="Save Plot as Image", command=save_plot_to_image)
save_plot_btn.grid(row=9, column=0, columnspan=2, pady=2)

# Matplotlib Figure
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=0, column=2, rowspan=12, padx=10, pady=5)

# Add the toolbar
toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
toolbar.update()
toolbar.grid(row=12, column=2, padx=10, pady=2, sticky="w")

# Results table using Treeview with scrollbars
table_frame = ttk.Frame(frame)
table_frame.grid(row=9, column=0, columnspan=2, pady=5, sticky="nsew")
frame.grid_rowconfigure(9, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

xscroll = ttk.Scrollbar(table_frame, orient="horizontal")
yscroll = ttk.Scrollbar(table_frame, orient="vertical")
results_table = ttk.Treeview(
    table_frame, columns=[], show="headings", height=15,
    xscrollcommand=xscroll.set, yscrollcommand=yscroll.set
)
xscroll.config(command=results_table.xview)
yscroll.config(command=results_table.yview)
results_table.grid(row=0, column=0, sticky="nsew")
xscroll.grid(row=1, column=0, sticky="ew")
yscroll.grid(row=0, column=1, sticky="ns")
table_frame.grid_rowconfigure(0, weight=1)
table_frame.grid_columnconfigure(0, weight=1)

# Variables to store last table for export
last_table = []
last_table_columns = []

def update_table_columns():
    if last_table_columns:
        results_table["columns"] = last_table_columns
        for col in last_table_columns:
            results_table.heading(col, text=col)
            results_table.column(col, width=120, minwidth=80, anchor="center")

# Update table columns after each solve
def after_plot_and_solve():
    update_table_columns()

# Patch plot_and_solve to call after_plot_and_solve
orig_plot_and_solve = plot_and_solve
def patched_plot_and_solve():
    orig_plot_and_solve()
    after_plot_and_solve()
plot_btn.config(command=patched_plot_and_solve)

# Initialize field visibility
update_fields()

# After you create fig, ax, and canvas:
zoom_factory(ax)
pan_factory(ax)

# After creating deriv_entry:
derive_btn = ttk.Button(frame, text="Derive", command=derive_function)
derive_btn.grid(row=4, column=2, sticky="w")

root.mainloop()
