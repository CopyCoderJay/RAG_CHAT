from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    """Extended user profile providing audit timestamps."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

