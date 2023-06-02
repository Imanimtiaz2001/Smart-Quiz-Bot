Smart Quiz Bot is an educational application designed to enhance learning and assess knowledge in English and Math. This Python-based bot generates personalized quizzes based on the user's knowledge level, identifies areas where the user needs improvement, and generates targeted quizzes to address those areas. It calculates scores based on user performance and provides a detailed analysis of time spent and question types.

Key Features Knowledge-Based Quizzes: The Smart Quiz Bot generates quizzes tailored to the user's knowledge level in English and Math. It ensures that questions are appropriate for the user's proficiency, allowing for effective learning and assessment.

Identifying Knowledge Gaps: By analyzing user performance, the bot identifies areas where the user lacks sufficient knowledge. This enables targeted assistance and provides opportunities for focused learning and improvement.

Personalized Quizzes: The system generates personalized quizzes specifically designed to address the identified knowledge gaps. This approach ensures that users receive relevant practice and reinforcement in areas where they need it the most.

Score Calculation: The Smart Quiz Bot calculates scores based on user performance in quizzes. This provides users with an objective measure of their progress and serves as a motivational tool for continuous learning.

Time and Question Type Analysis: Detailed analysis of time spent on each question and the distribution of question types enables users to identify patterns in their performance. This analysis assists in identifying specific question types in English and Math where users may require further practice or focus.

English and Math Quizzes: The bot offers two types of quizzes: English and Math. The English quizzes focus on meanings, and prepositions, and generate hints using the Natural Language Toolkit (NLTK). The Math quizzes cover addition, subtraction, multiplication, and division, and provide hints through related topic videos.

Question Clustering: Questions are automatically classified into clusters using the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. This clustering facilitates pattern recognition and mistake analysis, aiding users in understanding and addressing specific question types in English and Math.

Secure Login and Sign-Up: The Smart Quiz Bot implements personalized and secure login and sign-up features to ensure user privacy and data protection.

Implementation The Smart Quiz Bot is implemented using the Tkinter library for the graphical user interface (GUI) and incorporates Python for preprocessing and clustering of questions. The training set consists of 100 Math questions and 50 English questions, while the validation set includes 20 Math questions and 10 English questions. Each question is labeled with cluster labels, and true labels for model performance validation, and provides answer choices and options for users.

Getting Started To use the Smart Quiz Bot:

Install the required dependencies and libraries specified in the documentation.

Launch the application and create a personalized account with secure login credentials.

Access the English or Math quizzes based on your preference and knowledge level.

Complete the quizzes and receive scores, along with a detailed analysis of time spent and question types.

Utilize the personalized quizzes generated to improve knowledge in specific areas and track progress over time.

