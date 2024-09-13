# Initialize the logger
logging.basicConfig(level=logging.INFO)
def ocr_pdf_with_options(input_pdf_path: str, credentials_path: str):
    try:
        # # Read the input PDF file
        # with open(input_pdf_path, 'rb') as file:
        #     input_stream = file.read()


        # Read the input PDF file
        # with open(input_pdf_path, 'rb') as file:
        input_stream = input_pdf_path.read()


        
        # Load credentials
        with open(credentials_path, 'r') as f:
            credentials_data = json.load(f)
        client_id = credentials_data.get('client_credentials', {}).get('client_id')
        client_secret = credentials_data.get('client_credentials', {}).get('client_secret')
        # Set up credentials for the PDF service
        credentials = ServicePrincipalCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        # Create a PDF Services instance
        pdf_services = PDFServices(credentials=credentials)
        # Upload the input PDF file
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
        # Set OCR parameters
        ocr_pdf_params = OCRParams(
            ocr_locale=OCRSupportedLocale.EN_US,
            ocr_type=OCRSupportedType.SEARCHABLE_IMAGE
        )
        # Create an OCR job
        ocr_pdf_job = OCRPDFJob(input_asset=input_asset, ocr_pdf_params=ocr_pdf_params)
        # Submit the job and get the result
        location = pdf_services.submit(ocr_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, OCRPDFResult)
        # Get the resulting asset
        result_asset: CloudAsset = pdf_services_response.get_result().get_asset()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        # Return the output PDF stream
        return stream_asset.get_input_stream()
    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        logging.exception(f'Exception encountered while executing operation: {e}')
        return None
