from django.db import models

class AIInteraction(models.Model):
    """Model to track AI interactions for analytics"""
    user_query = models.TextField()
    ai_response = models.TextField()
    sources_used = models.JSONField(default=list, blank=True)
    processing_time = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"AI Interaction - {self.created_at}"

