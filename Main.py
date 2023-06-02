import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import json
import time
import csv
import random
import pandas as pd
import numpy as np
from transformers import pipeline, DistilBertTokenizer, DistilBertModel
from sklearn.cluster import DBSCAN
from tkinter import Tk, Label, Button
from PIL import Image
from PIL import ImageTk
from tkvideo import tkvideo
from PIL import ImageTk
import nltk
from nltk.corpus import wordnet
# load DistilBERT tokenizer and model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
class login(tk.Frame):
    credentials =  pd.read_csv("credentials.csv")
    def __init__(self,parent,controller):
        parent.configure(bg = "#fff")
        frame = tk.Frame.__init__(self,parent,bg = "white")
        #functions
        def onEnter(e):
            usernameEntry.delete(0,'end')
            
        def onLeave(e):
            name = usernameEntry.get()
            if name =='':
                usernameEntry.insert(0,'Username')
                
        def on_Enter(e):
            passwordEntry.delete(0,'end')
            
        def on_Leave(e):
            name = passwordEntry.get()
            if name =='':
                passwordEntry.insert(0,'Password')
                
        def login():
            credentials=pd.read_csv("credentials.csv")
            username = usernameEntry.get()
            password = passwordEntry.get()
            
            if credentials["username"].eq(username).any() and credentials["password"].eq(password).any():
               controller.show_frame(mainMenu)
            else:
                messagebox.showerror("Information","Invalid Username or Password!")
        #creating widgets
        heading = Label(self,text = 'Log In', fg = "black",bg="white",font = ('Ink Free',25,'bold'))
        usernameEntry = Entry(self, width = 25,fg = 'black',border = 0,bg='white',font = ('Ink Free',20))
        usernameEntry.insert(0,'Username')
        passwordEntry = Entry(self,show='*', width = 25,fg = 'black',border = 0,bg='white',font = ('Ink Free',20))
        passwordEntry.insert(0,'Password')
        enter = Button(self,width = 20,text='Enter',bg = '#57a1f8', fg = 'white',border = 0,font = ('Dubai',12,'bold'),command = login)
        signupPrompt = Label(self,text = "Don't have an account?", fg='black',bg ='white',font = ('Dubai',9))
        signUp = Button(self,width = 6,text='Sign Up',border = 0,bg='white',cursor = "hand2",fg = '#57a1f8',font = ('Dubai',9,"bold"),command = lambda: controller.show_frame(signup))


        #image
        decor1 = PhotoImage(file="decor1.PNG")
        decor2 = PhotoImage(file="decor2.PNG")
        decor3 = PhotoImage(file="decor3.PNG")
        decor4 = PhotoImage(file="decor4.PNG")
        decor5 = PhotoImage(file="decor5.PNG")
        decor6 = PhotoImage(file="decor6.PNG")

        #grid configure
        #self.columnconfigure(tuple(range(4)), weight=1)
        #self.rowconfigure(tuple(range(4)), weight=1)

        #placing widgets
        heading.grid(row=0,column=2)
        usernameEntry.grid(row=1,column=2)
        separator = ttk.Separator(orient="horizontal")
        #separator.place(in_ = usernameEntry, x=0, rely=1.0, height=3, relwidth=1.0)
        passwordEntry.grid(row=2,column=2)
        separatorPass = ttk.Separator(orient="horizontal")
        #separatorPass.place(in_ = passwordEntry, x=0, rely=1.0, height=3, relwidth=1.0)
        enter.grid(row = 3,column= 2)
        signupPrompt.place(in_ = enter,x=0, relx = -0.05,rely=1.3, height=9, relwidth=0.8)
        signUp.place(in_ = enter,x=0,relx = 0.7, rely=1.25, height=15, relwidth=0.3)

        d1= Label(self,image=decor1,bg="white")
        d1.image = decor1
        d1.grid(row=0,column=1)
        
        d2 = Label(self,image=decor2,bg="white")
        d2.image = decor2
        d2.grid(row=2,column=0)
        
        d3 = Label(self,image=decor3,bg="white")
        d3.image = decor3
        d3.grid(row=4,column=1)
        
        d4 = Label(self,image=decor4,bg="white")
        d4.image = decor4
        d4.grid(row=0,column=3)
        
        d5 = Label(self,image=decor5,bg="white")
        d5.image = decor5
        d5.grid(row=2,column=4)
        
        d6 = Label(self,image=decor6,bg="white")
        d6.image = decor6
        d6.grid(row=4,column=3)

        #binding functions
        usernameEntry.bind('<FocusIn>',onEnter)
        usernameEntry.bind('<FocusOut>',onLeave)
        passwordEntry.bind('<FocusIn>',on_Enter)
        passwordEntry.bind('<FocusOut>',on_Leave)
                
class signup(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent,bg="white")
        #functions
        def onEnter(e):
            usernameEntry.delete(0,'end')
            
        def onLeave(e):
            name = usernameEntry.get()
            if name =='':
                usernameEntry.insert(0,'Username')
                
        def on_Enter(e):
            passwordEntry.delete(0,'end')
            
        def on_Leave(e):
            name = passwordEntry.get()
            if name =='':
                passwordEntry.insert(0,'Password')
                
        def signup():
            credentials =  pd.read_csv("credentials.csv")
            username = usernameEntry.get()
            password = passwordEntry.get()
            
            if credentials["username"].eq(username).any():
                messagebox.showerror("Information","Username Taken!")
            else:
                with open('credentials.csv', mode='a',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([username, password])
                    file.close()
                    controller.show_frame(login)
            
        #creating widgets
        heading = Label(self,text = 'Sign Up', fg = "black",bg="white",font = ('Ink Free',25,'bold'))
        usernameEntry = Entry(self, width = 25,fg = 'black',border = 0,bg='white',font = ('Ink Free',20))
        usernameEntry.insert(0,'Username')
        passwordEntry = Entry(self, show='*',width = 25,fg = 'black',border = 0,bg='white',font = ('Ink Free',20))
        passwordEntry.insert(0,'Password')
        enter = Button(self,width = 20,text='Enter',bg = '#57a1f8', fg = 'white',border = 0,font = ('Dubai',12,'bold'),command = signup)
        
        #image
        decor1 = PhotoImage(file="decor2.PNG")
        decor2 = PhotoImage(file="decor3.PNG")
        decor3 = PhotoImage(file="decor1.PNG")
        decor4 = PhotoImage(file="decor6.PNG")
        decor5 = PhotoImage(file="decor5.PNG")
        decor6 = PhotoImage(file="decor4.PNG")
        imback = PhotoImage(file="back.PNG")
        #img_list = [decor1,decor2,decor3,decor4,decor5,decor6]
        
        #grid configure
        #self.columnconfigure(tuple(range(4)), weight=1)
        #self.rowconfigure(tuple(range(4)), weight=1)

        #placing widgets
        heading.grid(row=0,column=2)
        usernameEntry.grid(row=1,column=2)
        separator = ttk.Separator(orient="horizontal")
        #separator.place(in_ = usernameEntry, x=0, rely=1.0, height=3, relwidth=1.0)
        passwordEntry.grid(row=2,column=2)
        separatorPass = ttk.Separator(orient="horizontal")
        #separatorPass.place(in_ = passwordEntry, x=0, rely=1.0, height=3, relwidth=1.0)
        enter.grid(row = 3,column= 2)
        
        d1= Label(self,image=decor1,bg="white")
        d1.image = decor1
        d1.grid(row=0,column=1)
        
        d2 = Label(self,image=decor2,bg="white")
        d2.image = decor2
        d2.grid(row=2,column=0)
        
        d3 = Label(self,image=decor3,bg="white")
        d3.image = decor3
        d3.grid(row=4,column=1)
        
        d4 = Label(self,image=decor4,bg="white")
        d4.image = decor4
        d4.grid(row=0,column=3)
        
        d5 = Label(self,image=decor5,bg="white")
        d5.image = decor5
        d5.grid(row=2,column=4)
        
        d6 = Label(self,image=decor6,bg="white")
        d6.image = decor6
        d6.grid(row=4,column=3)
        
        back = Button(self,bg = 'white',image =imback,border = 0,command = lambda: controller.show_frame(login))
        back.image = imback
        back.place(anchor=NW)
        #binding functions
        usernameEntry.bind('<FocusIn>',onEnter)
        usernameEntry.bind('<FocusOut>',onLeave)
        passwordEntry.bind('<FocusIn>',on_Enter)
        passwordEntry.bind('<FocusOut>',on_Leave) 

class mainMenu(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent,bg="white",width = 800,height=450)
        global whichQuizChosen
        #functions      
        def englishChosen():
            frame = quiz(parent,controller,'EngTSet.json')
            controller.frames[quiz] = frame
            frame.grid(row =0, column = 0,sticky = "nsew")
            controller.show_frame(quiz)
        def mathChosen():
            frame = quiz(parent,controller,'MathsTSet.json')
            controller.frames[quiz] = frame
            frame.grid(row =0, column = 0,sticky = "nsew")
            controller.show_frame(quiz)
        #image
        immath = PhotoImage(file="math.PNG")
        imenglish = PhotoImage(file="english.PNG")
        imscience = PhotoImage(file="science.PNG")
        imhistory = PhotoImage(file="history.PNG")
        imurdu = PhotoImage(file="urdu.PNG")
        imislamic = PhotoImage(file="islamic.PNG")
        imback = PhotoImage(file="back.PNG")
        
        #creating widgets
        math = Button(self,bg = 'white',image =immath,border = 0,command = mathChosen)
        math.image = immath
        english = Button(self,bg = 'white',image =imenglish,border = 0,command = englishChosen)
        english.image = imenglish
        science = Button(self,bg = 'white',image =imscience,border = 0)
        science.image = imscience
        history = Button(self,bg = 'white',image =imhistory,border = 0)
        history.image = imhistory
        urdu = Button(self,bg = 'white',image =imurdu,border = 0)
        urdu.image = imurdu
        islamic = Button(self,bg = 'white',image =imislamic,border = 0)
        islamic.image = imislamic
        back = Button(self,bg = 'white',image =imback,border = 0,command = lambda: controller.show_frame(login))
        back.image = imback
        
        
        #placinf widgets
        back.place(anchor=NW)
        
        
        math.grid(row=0,column=0,padx=5,pady=60,ipadx=15,sticky=NE)
        english.grid(row=0,column=1,padx=5,pady=60,ipadx=15)
        science.grid(row=0,column=2,padx=5,pady=60,ipadx=15,sticky=NW)
        history.grid(row=1,column=0,padx=5,pady=5,ipadx=15,sticky=NE)
        urdu.grid(row=1,column=1,padx=5,pady=5,ipadx=15)
        islamic.grid(row=1,column=2,padx=5,pady=5,ipadx=15,sticky=NW)
        
        '''
        math.place(relx=0.2, rely=0.1,anchor= CENTER)
        english.place(relx=0.5, rely=0.1,anchor= CENTER)
        science.place(relx=0.8,rely=0.1,anchor= CENTER)
        history.place(relx=0.2,rely=0.5,anchor= CENTER)
        urdu.place(relx=0.5,rely=0.5,anchor= CENTER)
        islamic.place(relx=0.8,rely=0.5,anchor= CENTER)
        '''


def calculate_scores(tp, tn, fp, fn):
            accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
            precision = (tp / (tp + fp)) * 100
            recall = tp / (tp + fn) * 100
            f1_score = 2 * (precision * recall) / (precision + recall)
            return accuracy, precision, recall, f1_score

class quiz(tk.Frame):
    def __init__(self,parent,controller,whichQuizChosen):
        tk.Frame.__init__(self,parent,bg="white",width = 800,height=450)   
        # overall dataset, change name of dataset as necessary
        global model,tokenizer
        with open(whichQuizChosen, 'r') as f:
            data = json.load(f)
            filename = f.name
        # extract embeddings for each self.question to capture meaning
        q_embeddings = []
        for question in data['question']:
            # break up text into smaller units for processing (tokenizing)
            inputs = tokenizer(question, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

            # feed tokenized self.self.question into the DistilBERT model
            outputs = model(**inputs)

            # take mean of the embeddings across all self.self.question tokens
            # this results in a single vector that represents the meaning of the entire self.self.question
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            q_embeddings.append(embeddings)

        # cluster using dbscan clustering
        # decreasing eps and min_samples increases number of clusters
        # uses similarity in embeddings to cluster
            if filename == 'MathsTSet.json' or filename == 'MathsVSet.json' :
                dbscan = DBSCAN(eps=1.2, min_samples=2).fit(q_embeddings) # worked the best so far for maths
            elif filename == 'EngTSet.json' or filename == 'EngVSet.json':
                dbscan = DBSCAN(eps=2.1, min_samples=2).fit(q_embeddings) # worked the best so far for maths
        # group the questions by their clusters
        groupings = {
            "question": [],
            "cluster_label": [],
            "answer": [],
            "options": []
        }
        for i, label in enumerate(dbscan.labels_):
            # store self.questions belonging to the same cluster
            groupings["question"].append(data['question'][i])
            groupings["cluster_label"].append(str(label))
            groupings["answer"].append(data['answer'][i])
            groupings["options"].append(data['options'][i])

        # write groupings to a JSON file
        with open('clustered_questions.json', 'w') as outfile:
            json.dump(groupings, outfile, indent=4)

        if filename == 'MathsTSet.json':
            with open("MathsCTSet.json", "r") as f:
                math_data = json.load(f)

        if filename == 'MathsVSet.json':
            with open("MathsCVSet.json", "r") as f:
                math_data = json.load(f)
        # performance evaluation
        if filename == 'MathsTSet.json' or filename == 'MathsVSet.json':
        # load data from math file
        # create dictionary to store math results
            math_results = {}

        # loop through math clusters
            for cluster in set(math_data["cluster_label"]):

            # initialize counts to zero
                tp = 0
                tn = 0
                fp = 0
                fn = 0

            # loop through math questions
                for i, q in enumerate(math_data["question"]):

                # check if self.question is in current cluster
                    if math_data["cluster_label"][i] == cluster:

                    # check if self.question is correctly classified
                        if math_data["true_label"][i] == cluster:
                            tp += 1
                        else:
                            fp += 1

                # check if self.question is not in current cluster
                    else:

                    # check if self.question is correctly not classified
                        if math_data["true_label"][i] != cluster:
                            tn += 1
                        else:
                            fn += 1

            # add results to math dictionary
                math_results[f"cluster{cluster}_tp"] = tp
                math_results[f"cluster{cluster}_tn"] = tn
                math_results[f"cluster{cluster}_fp"] = fp
                math_results[f"cluster{cluster}_fn"] = fn
        if filename == 'EngTSet.json':
        # load data from english file
            with open("EngCTSet.json", "r") as f:
                eng_data = json.load(f)

        if filename == 'EngVSet.json':
        # load data from english file
            with open("EngCVSet.json", "r") as f:
                eng_data = json.load(f)
        if filename == 'EngTSet.json' or filename == 'EngVSet.json':
        # create dictionary to store english results
            eng_results = {}

        # loop through english clusters
            for cluster in set(eng_data["cluster_label"]):

            # initialize counts to zero
                tp = 0
                tn = 0
                fp = 0
                fn = 0

            # loop through english self.questions
                for i, q in enumerate(eng_data["question"]):

                # check if self.question is in current cluster
                    if eng_data["cluster_label"][i] == cluster:

                    # check if self.question is correctly classified
                        if eng_data["true__label"][i] == cluster:
                            tp += 1
                        else:
                            fp += 1

                # check if self.question is not in current cluster
                    else:

                    # check if self.question is correctly not classified
                        if eng_data["true__label"][i] != cluster:
                            tn += 1
                        else:
                            fn += 1

            # add results to english dictionary
                eng_results[f"cluster{cluster}_tp"] = tp
                eng_results[f"cluster{cluster}_tn"] = tn
                eng_results[f"cluster{cluster}_fp"] = fp
                eng_results[f"cluster{cluster}_fn"] = fn

        # open CSV file in write mode to clear the file and write from start
        with open('performance_eval.csv', mode='w', newline='') as results_file:
            results_writer = csv.writer(results_file)

            # write header row
            results_writer.writerow(['Language', 'Cluster', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            if filename == 'MathsTSet.json' or filename == 'MathsVSet.json':
                math_acc = 0
                math_prec = 0
                math_recall = 0
                math_f1 = 0
                for i in range(0, 4):
                    accuracy, precision, recall, f1_score = calculate_scores(math_results[f"cluster{i}_tp"], math_results[f"cluster{i}_tn"], math_results[f"cluster{i}_fp"], math_results[f"cluster{i}_fn"])
                # print(accuracy, precision, recall, f1_score)
                # add cluster-wise metrics to csv
                    results_writer.writerow(['Math', f'Cluster {i}', accuracy, precision, recall, f1_score])
                    math_acc += accuracy
                    math_prec += precision
                    math_recall += recall
                    math_f1 += f1_score

                math_acc_pc = math_acc / 4
                math_prec_pc = math_prec / 4
                math_recall_pc = math_recall / 4
                math_f1_pc = math_f1 / 4
            # print(math_acc_pc, math_prec_pc, math_recall_pc, math_f1_pc)

            # add math metrics to csv
                results_writer.writerow(['Math', 'Overall', math_acc_pc,  math_prec_pc, math_recall_pc, math_f1_pc])
            if filename == 'EngTSet.json' or filename == 'EngVSet.json':
                eng_acc = 0
                eng_prec = 0
                eng_recall = 0
                eng_f1 = 0
                for i in range(0, 2):
                    accuracy, precision, recall, f1_score = calculate_scores(eng_results[f"cluster{i}_tp"], eng_results[f"cluster{i}_tn"], eng_results[f"cluster{i}_fp"],eng_results[f"cluster{i}_fn"])
                # print(accuracy, precision, recall, f1_score)
                # add cluster-wise metrics to csv
                    results_writer.writerow(['English', f'Cluster {i}', accuracy, precision, recall, f1_score])
                    eng_acc += accuracy
                    eng_prec += precision
                    eng_recall += recall
                    eng_f1 += f1_score

                eng_acc_pc = eng_acc / 2
                eng_prec_pc = eng_prec / 2
                eng_recall_pc = eng_recall / 2
                eng_f1_pc = eng_f1 / 2
            # print(eng_acc_pc, eng_prec_pc, eng_recall_pc, eng_f1_pc)

            # add english metrics to csv
                results_writer.writerow(['English', 'Overall', eng_acc_pc, eng_prec_pc, eng_recall_pc, eng_f1_pc])
                
            ###########################################################################################################
            ###########################################################################################################
            ###########################################################################################################
            ###########################################################################################################
            
            
        with open('clustered_questions.json') as f:
            data = json.load(f)

        # Find unique cluster labels (excluding -1)
        unique_labels = set(data['cluster_label'])
        unique_labels.discard("-1")

        # Select 2 random self.questions from each unique cluster label
        selected_questions = []
        for label in unique_labels:
            self.questions_in_cluster = [q for q, l in zip(data['question'], data['cluster_label']) if l == label]
            
            if filename == 'MathsTSet.json' or filename == 'MathsVSet.json':
            
                selected_questions.extend(random.sample(self.questions_in_cluster, 2)) # for maths

            elif filename == 'EngTSet.json'or filename == 'EngVSet.json':
            
                selected_questions.extend(random.sample(self.questions_in_cluster, 4)) # for maths

        
        # Create a new dictionary with the selected self.questions, options, and answers
        selected_data = {'question': [], 'options': [], 'answer': [], 'cluster_label': []}
        for self.question, options, answer, cluster_label in zip(data['question'], data['options'], data['answer'], data['cluster_label']):
            if self.question in selected_questions:
                selected_data['question'].append(self.question)
                selected_data['options'].append(options)
                selected_data['answer'].append(answer)
                selected_data['cluster_label'].append(cluster_label)

        # Use the selected data to create the quiz
        self.question = selected_data['question']
        self.options = selected_data['options']
        self.answer = selected_data['answer']
        cluster_label = selected_data['cluster_label']
       
        self.isQuiz= True
        self.quizChosen = whichQuizChosen
        self.q_no = 0
        self.display_title()
        self.display_question()
        self.opt_selected = IntVar()
        self.opts = self.radio_buttons()
        self.display_options()
        self.data_size = len(self.question)
        self.buttons()
        self.correct = 0
        self.start_time = time.time()
        self.results = []
        self.cluster_labels = cluster_label
        
        
    def display_result(self):
        end_time = time.time()  # End time for the quiz
        quiz_time = round(end_time - self.start_time, 3)  # Total time taken for the quiz, rounded to 3 decimal places
        quiz_time_str = time.strftime("%H:%M:%S", time.gmtime(quiz_time))  # Convert quiz time to HH:MM:S format
        wrong_count = self.data_size - self.correct
        correct = f"Correct: {self.correct}"
        wrong = f"Incorrect: {wrong_count}"
        score = int(self.correct / self.data_size * 100)
        result = f"Score: {score}%\nTime taken: {quiz_time_str}"
        messagebox.showinfo("Result", f"{result}\n{correct}\n{wrong}")
        
        print(self.results)
        with open('quiz_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Question', 'Result', 'Time Taken', 'Cluster Label'])
            for i, res in enumerate(self.results):
                time_str = time.strftime("%S", time.gmtime(res[1]))  # Convert time to S format
                cluster_label = self.cluster_labels[i]
                writer.writerow([i + 1, res[0], time_str, cluster_label])

        # display on GUI
        if wrong_count > 0:
            for widgets in self.winfo_children():
                widgets.destroy()
            
            titlebg = Label(self,width=200,height = 3, bg=self.whichColor())
            titlebg.place(x=0, y=2)
            
            titletext ="Review wrong answers!"
            title = Label(self, text=titletext,bg=self.whichColor(),fg = "white",font=('Ink Free',25,'bold'), anchor='w')
            title.place(x=240, y=5)
            ypos = 80
            for i, res in enumerate(self.results):
                if res[0] == "Incorrect":
                    if self.quizChosen == 'MathsVSet.json':
                        question = "Question "+ str(i + 1)+": What is the answer of "+ self.question[i]
                    else: question = "Question "+ str(i + 1)+": "+ self.question[i]
                    q_no = Label(self, text=question, width=60, bg = "white",font=('Dubai',12,'bold'), anchor='w')
                    q_no.place(x=60, y = ypos)
                    answer = "Correct answer: " + (self.options[i][self.answer[i] - 1])
                    answerlabel = Label(self, text=answer, width=60, bg = "white",font=('Dubai',12), anchor='w')
                    answerlabel.place(x = 470,y = ypos+2)
                    if '\n' in question:
                        ypos+=52
                    else:ypos+=30
                    self.buttons()
        else: self.destroy()


   
    def back_btn(self):
        if(self.q_no != 0):
            self.q_no -= 1
            if (self.results[self.q_no])[0] == 'Correct':
                self.correct-=1
                print("hello")
            self.results.pop()
            self.display_question()
            self.display_options()
            self.buttons()        
            

    def next_btn(self):
        self.check_ans(self.q_no)
        self.q_no += 1
        if self.q_no == self.data_size:
            self.display_result()
            #self.destroy()
        else:
            self.display_question()
            self.display_options()
            self.buttons()
    
    
    def displayRetestQs(self):
        # now analyzing results and getting retest questions
        # Read quiz_results.csv file
        df = pd.read_csv("quiz_results.csv")

        # check the rows for which the answer is incorrect or took more than 30 secs to retest the questions
        if ((df["Result"] == "Incorrect").any()) or ((df["Time Taken"] > 30).any()):
            for widgets in self.winfo_children():
                widgets.destroy()

        # Filter quiz_results.csv for questions either attempted incorrectly or took more than 30 seconds
            filtered_df = df[(df["Result"] == "Incorrect") | (df["Time Taken"] > 30)]

        # Get the unique cluster labels for the filtered questions
            retest_cluster_labels = filtered_df["Cluster Label"].unique().tolist()

        # Read clustered_questions.json file
            with open("clustered_questions.json") as f:
                clustered_questions = json.load(f)

        # Get one question per cluster label to retest
            new_questions = {
                "question": [],
                "options": [],
                "answer": [],
                "cluster_label": []
            }
            
            if self.quizChosen == 'MathsTSet.json' or self.quizChosen == 'MathsVSet.json':
                for label in retest_cluster_labels:
                    question_indices = [i for i, l in enumerate(clustered_questions["cluster_label"]) if int(l) == label]
                    new_question_index = random.choice(question_indices)  # Choose a random question from the cluster
                    new_question = {
                        "question": clustered_questions["question"][new_question_index],
                        "options": clustered_questions["options"][new_question_index],
                        "answer": clustered_questions["answer"][new_question_index],
                        "cluster_label": clustered_questions["cluster_label"][new_question_index]
                }
                    new_questions["question"].append(new_question["question"])
                    new_questions["options"].append(new_question["options"])
                    new_questions["answer"].append(new_question["answer"])
                    new_questions["cluster_label"].append(new_question["cluster_label"])

        # Store the new questions in a JSON file
                with open("retest_questions.json", "w") as f:
                    json.dump(new_questions, f)

        # for english
            elif self.quizChosen == 'EngTSet.json'or self.quizChosen == 'EngVSet.json':
                for label in retest_cluster_labels:
                # Get all questions for this cluster label
                    question_indices = [i for i, l in enumerate(clustered_questions["cluster_label"]) if int(l) == label]
                # Count the number of questions answered incorrectly for this cluster label
                    incorrect_questions = filtered_df[filtered_df["Cluster Label"] == label]
                    incorrect_count = len(incorrect_questions.index)
                # Choose the number of questions to store based on the number answered incorrectly
                    num_questions = 1 if incorrect_count <= 2 else 2
                # Choose random questions from the cluster
                    new_question_indices = random.sample(question_indices, num_questions)
                    for index in new_question_indices:
                        new_question = {
                            "question": clustered_questions["question"][index],
                            "options": clustered_questions["options"][index],
                            "answer": clustered_questions["answer"][index],
                            "cluster_label": clustered_questions["cluster_label"][index]
                    }
                        new_questions["question"].append(new_question["question"])
                        new_questions["options"].append(new_question["options"])
                        new_questions["answer"].append(new_question["answer"])
                        new_questions["cluster_label"].append(new_question["cluster_label"])

            # Store the new questions in a JSON file
                with open("retest_questions.json", "w") as f:
                    json.dump(new_questions, f)
                    f.flush()  # ensure that all data is written to disk

            # now open retest_questions.json and store in relevant variables to send to Quiz object
            with open('retest_questions.json') as f:
                data_r = json.load(f)
            self.question = (data_r['question'])
            self.options = (data_r['options'])
            self.answer = (data_r['answer'])
            cluster_label = (data_r['cluster_label'])

            self.isQuiz = False
            self.q_no = 0
            self.display_title()
            self.display_question()
            self.opt_selected = IntVar()
            self.opts = self.radio_buttons()
            self.display_options()
            self.data_size = len(self.question)
            self.buttons()
            self.correct = 0
            self.start_time = time.time()
            self.results = []
            self.cluster_labels = cluster_label
    
    def whichColor(self):
        if self.quizChosen == 'MathsVSet.json' or self.quizChosen == 'MathsTSet.json':
            return '#eb402d'
        else: return'#4f9bda'
        
    def buttons(self):
        prev = PhotoImage(file="previous.PNG")
        next = PhotoImage(file="next.PNG")
        main = PhotoImage(file="mainMenu.PNG")
    
        if self.q_no != self.data_size:
            next_button = Button(self, image = next,bg = self.whichColor(), command=self.next_btn,border = -1)
            next_button.image = next
            next_button.place(x=440, y=340)
        else: 
            if self.isQuiz:
                next_button = Button(self, image = next,bg = self.whichColor(), command=self.displayRetestQs,border = -1)
                next_button.image = next
                next_button.place(x=640,y=390)
            
        
        if self.q_no != 0 and self.q_no != self.data_size:
            back_button = Button(self, image = prev,bg = self.whichColor(),command=self.back_btn,border = -1)
            back_button.image =prev
            back_button.place(x=240, y=340)
        
        quit_button = Button(self, image = main ,bg = self.whichColor(),command=self.destroy,border = 0)
        quit_button.image = main
        quit_button.place(x=730,y=10)

    def display_options(self):
        val = 0
        self.opt_selected.set(0)
        if self.q_no < len(self.options):
            for option in self.options[self.q_no]:
                self.opts[val]['text'] = option
                val += 1
    def get_definition(self, query):
        synsets = wordnet.synsets(query)
        if synsets:
            definition = synsets[0].definition()
            return definition
        else:
            return None
        
    def hint(self):
        if self.quizChosen == 'MathsVSet.json' or self.quizChosen == 'MathsTSet.json':
            print(self.cluster_labels[self.q_no])
            if self.cluster_labels[self.q_no] == "0":
                video_path = "0.mp4"
            elif self.cluster_labels[self.q_no] == "1":
                video_path = "1.mp4" 
            elif self.cluster_labels[self.q_no] == "2":
                video_path = "2.mp4" 
            elif self.cluster_labels[self.q_no] == "3":
                video_path = "3.mp4" 
            else: 
                ("hello")          
            my_label = Label(application)
            my_label.pack()
            my_label.place(x=0, y=0)  # Position the label at (0, 0) within the window
            video_player = tkvideo(video_path, my_label, loop=0, size=(800, 450))
            video_player.play()
            application.mainloop()
        else:
            gui = Tk()
            gui.geometry("800x450")
            gui.title("Hint")
            titlebg = Label(gui, width=200, height=3, bg="blue")
            titleText = "Explanation"
            title = Label(gui, text=titleText, bg="blue", fg="white", font=('Ink Free', 25, 'bold'))
            titlebg.place(x=0, y=2)
            title.place(x=360, y=5)
            if self.cluster_labels[self.q_no]=="0":
                query = (self.options[self.q_no][self.answer[self.q_no] - 1])
                nltk.download('wordnet')
                definition = "Meaning is '"+self.get_definition(query)+"'."
                if definition:
                    textx = definition
                    q_no = Label(gui, text=textx, width=80, height=2,
                                 font=('Dubai', 15, 'bold'), anchor='w')
                    q_no.place(x=230, y=80)
                else:
                    textx = "Definition does not exist."
                    q_no = Label(gui, text=textx, width=60, height=2,
                                 font=('Dubai', 15, 'bold'), anchor='w')
                    q_no.place(x=230, y=80)

            else:
                # Load the JSON data from the file
                with open('prep_ex.json') as file:
                    data = json.load(file)

                # Function to access examples by preposition
                def get_examples(preposition):
                    if preposition in data:
                        return data[preposition]
                    else:
                        return []


                examples = get_examples(self.options[self.q_no][self.answer[self.q_no] - 1])

                if examples:
                    example_txt=f"Sentence usages for '{self.options[self.q_no][self.answer[self.q_no] - 1]}':"
                    for example in examples:
                        example_txt=example_txt+"\n" + example
                    textx = example_txt
                    q_no = Label(gui, text=textx, width=80, height=7,
                                 font=('Dubai', 15, 'bold'), anchor='w')
                    q_no.place(x=230, y=80)
                else:
                    print(f"No examples found for '{preposition}'")
            gui.mainloop()
    def display_question(self):
        if self.quizChosen == 'MathsVSet.json':
            textx = str(self.q_no + 1) + ') What is the answer of ' + self.question[self.q_no]
        else: textx = str(self.q_no + 1) + ') ' + self.question[self.q_no]
        q_no = Label(self, text=textx, width=60,height = 2, bg = "white",
        font=('Dubai',15,'bold'), anchor='w')
        if not self.isQuiz:
            hintim = PhotoImage(file="hint.PNG")
            ## if need to pass a argument to hint window add argument to def hint(self,argument) and command = self.hint(argument) 
            hint = Button(self, image = hintim, bg = "white", command = self.hint,anchor='w',border = 0)
            hint.image = hintim
            hint.place(x=650, y=100)
        q_no.place(x=230, y=80)
        

    def display_title(self):
        titlebg = Label(self,width=200,height = 3, bg=self.whichColor())
        if self.isQuiz:
            titleText = "Quiz"
        else:
            titleText = "Re Test"
        title = Label(self, text=titleText,bg=self.whichColor(), fg="white", font=('Ink Free',25,'bold'))
        titlebg.place(x=0, y=2)
        title.place(x = 360,y = 5)

    def radio_buttons(self):
        q_list = []
        y_pos = 140
        while len(q_list) < 4:
            radio_btn = Radiobutton(self, text=" ", variable=self.opt_selected, bg = "white",
            value=len(q_list)+1, font=("Dubai", 13))
            q_list.append(radio_btn)
            radio_btn.place(x=270, y=y_pos)
            y_pos += 40
        return q_list

    def check_ans(self, q_no):
            start_time = time.time() # Start time for the question
            if self.opt_selected.get() == self.answer[q_no]:
                result = 'Correct'
                self.correct += 1
            else:
                result = 'Incorrect'
            end_time = time.time() # End time for the question
            time_taken = (end_time - start_time)*(10**5) # Time taken for the question
            self.results.append((result, time_taken))
            print('start: '+str(start_time)+'end: '+str(end_time)+'time_taken'+str(time_taken))
            return result
            

    
        
class app(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        window = tk.Frame(self)
        window.title = "Smart Quiz"
        window.pack(side = "top",fill = "both",expand = True)  
        window.grid_rowconfigure(0,weight = 1,minsize=4)
        window.grid_columnconfigure(0,weight =1,minsize=4)
           
        self.frames = {}
        for F in (login,signup,mainMenu):
            frame = F(window,self)
            
            self.frames[F] = frame
            frame.grid(row =0, column = 0,sticky = "nsew")

        self.show_frame(login)
        
    def show_frame(self,page):
            frame = self.frames[page]
            frame.tkraise()
            
application = app()
application.mainloop()
