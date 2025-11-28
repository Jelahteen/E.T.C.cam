# core/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm # Import AuthenticationForm
from django.contrib.auth import get_user_model

User = get_user_model()

class SimpleUserCreationForm(UserCreationForm):
    """
    A custom user creation form where the email is used as the username.
    """
    email = forms.EmailField(
        label="Email",
        max_length=254,
        widget=forms.EmailInput(attrs={'autocomplete': 'email'})
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email",) # Ensure 'username' is also included so it's handled by UserCreationForm logic, then overridden.
        # It's crucial to still pass 'username' here so UserCreationForm processes it.
        # We'll set the initial username in the view if needed, or rely on clean_email to set it.
        # However, for saving, the save method below is primary.

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email is already used.")
        return email

    def save(self, commit=True):
        # Override the save method to set the username as the email
        user = super().save(commit=False)
        user.username = self.cleaned_data['email'] # Ensure username is set to email
        if commit:
            user.save()
        return user

class LoginForm(AuthenticationForm):
    """
    A custom login form based on Django's built-in AuthenticationForm.
    This is added to resolve the ImportError in views.py.
    """
    # No custom fields needed here unless you want specific widgets or validation
    pass

