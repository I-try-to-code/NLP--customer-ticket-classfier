import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini 3.5 Flash for newer API keys
        self.model = genai.GenerativeModel('gemini-3.5-flash')
        
    def generate_response(self, ticket_text: str, queue: str, priority: str) -> str:
        """
        Generates a customer support response using Gemini 2.5 Flash.
        """
        prompt = f"""
        You are an expert customer support agent for a SaaS enterprise.
        A user has submitted a support ticket.
        
        Ticket Details:
        - Assigned Queue: {queue}
        - Priority Level: {priority}
        
        Original Ticket:
        "{ticket_text}"
        
        Task: Write a professional, empathetic, and helpful response to the customer. 
        - Acknowledge their issue clearly.
        - Set proper expectations based on the priority level.
        - Do not use placeholders like [Your Name]. Just sign off as "Customer Support Team".
        - Keep the formatting clean and easy to read.
        """
        
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    service = LLMService()
    print(service.generate_response("My app keeps crashing when I upload photos.", "Technical Support", "High"))
