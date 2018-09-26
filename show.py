import tkinter as tk
from tkinter import font
import tensorflow as tf
import numpy as np
import os
import onepic
#import cv2
#loc = "C:/Users/54741/Desktop/picreg/three/test/cat.4082.jpg"


def clear_entries():
    Query.delete(0,'end')
    Result.delete(0,'end')

def show_input():

    query_w = Query.get()
    command = query_w

    res = output(command)

    Result.delete(0,'end')
    Result.insert(0, res)




main = tk.Tk()

fnt = font.Font(size=18)

main.title("Relative Networks")
main.geometry('600x200')
tk.Label(main, text = "Query:",height=1,width=5).grid(row=0)
tk.Label(main, text = "Result:",height=5,width=5).grid(row=1)

Query = tk.Entry(main,width=300,font=fnt)
Result = tk.Entry(main,width=300,font=fnt)

Query.grid(row=0, column=1)
Result.grid(row=1, column=1)

Query.place(x=50,width=900)
Result.place(x=50,y=50,width=900)

tk.Button(main, text='Clear', command=clear_entries).grid(row=4, column=0, pady=4)
tk.Button(main, text='Query', command=show_input).grid(row=4, column=1, pady=4)

tk.mainloop()
