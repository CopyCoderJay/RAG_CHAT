from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="document",
            old_name="supabase_id",
            new_name="external_id",
        ),
    ]

