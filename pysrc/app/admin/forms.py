from flask_security.forms import LoginForm as BaseLoginForm

class LoginForm(BaseLoginForm):
    def validate(self, extra_validators=None):
        return super(LoginForm, self).validate()