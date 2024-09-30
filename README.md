# Sentiment Analysis System 


# Introduction
The Sentiment Analysis System classifies textual data (e.g., product reviews, social media posts, customer feedback) into Positive, Negative, or Neutral categories. 
It uses a microservices architecture to handle data ingestion, text preprocessing, sentiment classification, and result storage, with each service running in a separate Docker container for easy deployment, management, and scalability.


# Features
This application automatically categorizes text into Positive, Negative, or Neutral sentiments.
It helps businesses understand customer sentiment, leading to better decision-making and saves time & resources compared to manual classification.



# Installation
It's a web based system where an audio data will be uploaded as input.
Application will detect the sentiment and will categorize the result into Positive, Negative, or Neutral as output.


# Usage
Businesses can use this application to automatically classify customer feedback, social media posts, and product reviews into Positive, Negative, or Neutral sentiments. 
It stores these classifications and allows querying by time, text, or sentiment type for valuable insights.


# System Design
Microservices: Independent services communicate via REST APIs.
Containerization: Docker ensures platform independence.
Scalability: Horizontally scalable, with future Kubernetes orchestration.
API Gateway: (Future Scope) Manage traffic and secure communication.


# Contact
For any questions or concerns, please reach out to:

Saumya Mitra - m23aid010@iitj.ac.in
Arka Gayen - m23aid016@iitj.ac.in
Debadrita Biswas - m23aid041@iitj.ac.in