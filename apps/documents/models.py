from django.db import models
from apps.chat.models import Chat

class Document(models.Model):
    """Document model for PDF files."""
    external_id = models.CharField(max_length=36, unique=True, null=True, blank=True)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='documents')
    filename = models.CharField(max_length=255)
    file_path = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename

class DocumentChunk(models.Model):
    """Document chunk model for vector storage"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    chunk_id = models.IntegerField()
    page_number = models.IntegerField()
    content = models.TextField()
    embedding = models.JSONField()  # Store vector embeddings
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['document', 'chunk_id']
    
    def __str__(self):
        return f"{self.document.filename} - Chunk {self.chunk_id}"

