from flask import Flask
from flask_wtf import FlaskForm, RecaptchaField
from wtforms import (
    DateField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import URL, DataRequired, Email, EqualTo, Length


class DataForm(FlaskForm):
    data_format = SelectField("data_format", [DataRequired()], 
                        choices=[
                            "None", 
                            "Table",
                            "Audio",
                            "Image",
                            "Text",
                            "Time Series"
                            ])
    graph_vertex = StringField(
        "graph_vertex", [DataRequired()]
    )
    errors = SelectField("errors", choices=["No", "Yes"])
    submit = SubmitField("Submit")



app.config.from_object("config.Config")

with app.app_context():
    from . import routes

