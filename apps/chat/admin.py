from django.contrib import admin
from .models import Chat, Message

@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'external_id', 'created_at')
    list_filter = ('created_at', 'user')
    search_fields = ('title', 'user__username')
    readonly_fields = ('external_id', 'created_at', 'updated_at')

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('chat', 'role', 'content_preview', 'created_at')
    list_filter = ('role', 'created_at', 'chat__user')
    search_fields = ('content', 'chat__title')
    readonly_fields = ('created_at',)
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content Preview'

