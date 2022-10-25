import tkinter as tk


g = "elo"

text = "The following network is " + g.upper() + \
       ". \n Please click 'Post to python' in the browser when investigated." + \
       "\n Default name for the graph is: " + g + "_" + str(10)
window = tk.Tk()
lbl = tk.Label(window, text="Input")
lbl.pack()
txt = tk.Text(window, width=100, height=20)
txt.pack()
txt.insert("1.0", text)
button = tk.Button(window, text="Show", command=window.destroy)
button.pack()
window.mainloop()
