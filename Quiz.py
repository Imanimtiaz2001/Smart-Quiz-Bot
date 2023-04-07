from tkinter import *
from tkinter import messagebox as mb
import json
import time
import csv
class Quiz:
	def __init__(self):
		self.q_no=0
		self.display_title()
		self.display_question()
		self.opt_selected=IntVar()
		self.opts=self.radio_buttons()
		self.display_options()
		self.buttons()
		self.data_size=len(question)
		self.correct=0
		self.start_time = time.time() 
		self.results = []
	def display_result(self):
	    end_time = time.time()  # End time for the quiz
	    quiz_time = round(end_time - self.start_time, 3)  # Total time taken for the quiz, rounded to 3 decimal places
	    quiz_time_str = time.strftime("%M:%S:%f", time.gmtime(quiz_time))[:-3]  # Convert quiz time to MM:SS:MS format
	    wrong_count = self.data_size - self.correct
	    correct = f"Correct: {self.correct}"
	    wrong = f"Incorrect: {wrong_count}"
	    score = int(self.correct / self.data_size * 100)
	    result = f"Score: {score}%\nTime taken: {quiz_time_str}"
	    mb.showinfo("Result", f"{result}\n{correct}\n{wrong}")
	    with open('quiz_results.csv', mode='w', newline='') as file:
	        writer = csv.writer(file)
	        writer.writerow(['Question', 'Result', 'Time Taken'])
	        for i, res in enumerate(self.results):
	            time_str = time.strftime("%M:%S:%f", time.gmtime(res[1]))[:-3]  # Convert time to MM:SS:MS format
	            writer.writerow([i+1, res[0], time_str])
	def next_btn(self):
		if self.check_ans(self.q_no)==True:
			self.correct += 1	
		self.q_no += 1
		if self.q_no==self.data_size:
			self.display_result()
			gui.destroy()
		else:
			self.display_question()
			self.display_options()
	def buttons(self):
		next_button = Button(gui, text="Next",command=self.next_btn,
		width=10,bg="blue",fg="white",font=("ariel",16,"bold"))
		next_button.place(x=350,y=380)
		quit_button = Button(gui, text="Quit", command=gui.destroy,
		width=5,bg="black", fg="white",font=("ariel",16," bold"))
		quit_button.place(x=700,y=50)
	def display_options(self):
		val = 0
		self.opt_selected.set(0)
		if self.q_no < len(options):
			for option in options[self.q_no]:
				self.opts[val]['text'] = option
				val += 1
	def display_question(self):
		q_no = Label(gui, text=question[self.q_no], width=60,
		font=( 'ariel' ,16, 'bold' ), anchor= 'w' )
		q_no.place(x=70, y=100)
	def display_title(self):
		title = Label(gui, text="GeeksforGeeks QUIZ",
		width=50, bg="green",fg="white", font=("ariel", 20, "bold"))
		title.place(x=0, y=2)
	def radio_buttons(self):
		q_list = []
		y_pos = 150
		while len(q_list) < 4:
			radio_btn = Radiobutton(gui,text=" ",variable=self.opt_selected,
			value = len(q_list)+1,font = ("ariel",14))
			q_list.append(radio_btn)
			radio_btn.place(x = 100, y = y_pos)
			y_pos += 40
		return q_list
	def check_ans(self, q_no):
            start_time = time.time() # Start time for the question
            if self.opt_selected.get() == answer[q_no]:
                result = 'Correct'
                self.correct += 1
            else:
                result = 'Incorrect'
            end_time = time.time() # End time for the question
            time_taken = round(end_time - start_time) # Time taken for the question
            self.results.append((result, time_taken))
            return result
gui = Tk()
gui.geometry("800x450")
gui.title("GeeksforGeeks Quiz")
with open('data1.json') as f:
    data = json.load(f)
question = (data['question'])
options = (data['options'])
answer = (data[ 'answer'])
quiz = Quiz()
gui.mainloop()
