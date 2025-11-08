from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="chat",
            old_name="supabase_id",
            new_name="external_id",
        ),
        migrations.RemoveField(
            model_name="message",
            name="supabase_id",
        ),
    ]

