import logging
import smtplib
from email.message import EmailMessage
from typing import Union
import pandas as pd

# We import your configuration singleton based on your docstring path
from config.settings import get_settings

logger = logging.getLogger(__name__)

def format_alert_html(alerts_df: pd.DataFrame, regime: str, run_date: str) -> str:
    """
    Formats an alerts DataFrame into a clean HTML email.
    Separated from the sending logic to facilitate unit testing.
    """
    # Convert the DataFrame to a basic HTML table with some styling
    table_html = alerts_df.to_html(
        index=False, 
        classes="alert-table", 
        float_format=lambda x: f"{x:.2f}"
    )
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .header {{ background-color: #f44336; color: white; padding: 10px; text-align: center; }}
            .info {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #2196F3; }}
            .alert-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            .alert-table th, .alert-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .alert-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚠️ Agricultural Commodities Alert</h2>
        </div>
        
        <div class="info">
            <p><strong>Execution date:</strong> {run_date}</p>
            <p><strong>Detected market regime:</strong> {regime}</p>
            <p>The following signals have been detected with an absolute Z-Score exceeding the configured threshold:</p>
        </div>

        {table_html}
        
        <br>
        <p><small>This is an automated message from the agricultural data pipeline.</small></p>
    </body>
    </html>
    """
    return html


def send_alert(signals: Union[pd.DataFrame, list, dict], regime: str, run_date: str) -> None:
    """
    Filters signals based on the z-score threshold and sends an HTML email.
    If SMTP is not configured, it logs the content instead.
    """
    settings = get_settings()
    
    # 1. Safely retrieve alert configurations (with fallbacks)
    threshold = getattr(settings, 'alert_zscore_threshold', 2.0)
    smtp_server = getattr(settings, 'smtp_server', None)
    smtp_port = getattr(settings, 'smtp_port', 587)
    smtp_user = getattr(settings, 'smtp_username', None)
    smtp_pass = getattr(settings, 'smtp_password', None)
    email_to = getattr(settings, 'alert_email_to', 'admin@example.com')
    email_from = getattr(settings, 'alert_email_from', 'pipeline@example.com')

    # 2. Ensure signals is a DataFrame
    if not isinstance(signals, pd.DataFrame):
        signals = pd.DataFrame(signals)
        
    if 'z_score' not in signals.columns:
        logger.error("The 'z_score' column was not found in the signals data. Alert canceled.")
        return

    # 3. Filter signals that exceed the threshold (absolute)
    alerts_df = signals[signals['z_score'].abs() > threshold]

    if alerts_df.empty:
        logger.info(f"No signals with an absolute z-score greater than {threshold}. No alert will be sent.")
        return

    # 4. Generate HTML content
    html_content = format_alert_html(alerts_df, regime, run_date)

    # 5. Sending logic or fallback to log
    if not smtp_server:
        logger.warning(
            "SMTP server not configured in Settings. "
            "Logging the alert content instead of sending an email:"
        )
        logger.info(f"\n--- HTML ALERT START ---\n{html_content}\n--- HTML ALERT END ---")
        return

    # 6. Sending the email via SMTP
    msg = EmailMessage()
    msg['Subject'] = f"Agricultural Pipeline Alert - {run_date} (Regime: {regime})"
    msg['From'] = email_from
    msg['To'] = email_to
    msg.set_content("Your email client does not support HTML.")
    msg.add_alternative(html_content, subtype='html')

    try:
        logger.info(f"Sending email alert to {email_to} via {smtp_server}...")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("Email alert successfully sent.")
        
    except Exception as e:
        logger.error(f"Failed to send the alert email: {e}")
        logger.info(f"Fallback to log - HTML Content:\n{html_content}")