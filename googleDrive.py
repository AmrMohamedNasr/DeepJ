from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from apiclient import errors
from oauth2client.client import GoogleCredentials
import tempfile

colab_run = True

class GoogleDriveInterface:
    def __init__(self):
        gauth = GoogleAuth()
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            if (not colab_run):
                # Authenticate if they're not there
                gauth.LocalWebserverAuth()
            else:
                from google.colab import auth
                auth.authenticate_user()
                gauth.credentials = GoogleCredentials.get_application_default()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile("mycreds.txt")
        self.drive = GoogleDrive(gauth)
        self.creds = gauth

    def uploadFile(self, path, folder_id="'1xOe202CuYu5VpPsRitoT0jFulHDSUnPh'", binary=False):
        file_name = os.path.basename(path)
        print('Uploading ', file_name)
        if (binary):
            with open(path,"rb") as file:
                data = file.read()
                with tempfile.NamedTemporaryFile(delete=False) as source_file:
                    source_file.write(data)
                    data = source_file.name
                children = self.drive.ListFile({'q': folder_id + " in parents and trashed=false"}).GetList()
                for child in children:
                    if (child['originalFilename'] == file_name):
                        child.SetContentFile(data)
                        child.Upload()
                        print('Updated ', file_name)
                        return
                file_drive = self.drive.CreateFile({'title': file_name, "parents":  [{"id": folder_id[1:-1]}]  })  
                file_drive.SetContentFile(data)
                file_drive.Upload()
                print('Uploaded ', file_name)
        else:
            with open(path,"r") as file:
                data = file.read()
                children = self.drive.ListFile({'q': folder_id + " in parents and trashed=false"}).GetList()
                for child in children:
                    if (child['originalFilename'] == file_name):
                        child.SetContentString(data)
                        child.Upload()
                        print('Updated ', file_name)
                        return
                file_drive = self.drive.CreateFile({'title': file_name, "parents":  [{"id": folder_id[1:-1]}]  })  
                file_drive.SetContentString(data)
                file_drive.Upload()
                print('Uploaded ', file_name)
    def downloadFile(self, path, folder_id="'1xOe202CuYu5VpPsRitoT0jFulHDSUnPh'", binary=False):
        file_name = os.path.basename(path)
        print('Requesting ', file_name)
        children = self.drive.ListFile({'q': folder_id + " in parents and trashed=false"}).GetList()
        for child in children:
            if (child['originalFilename'] == file_name):
                file = self.drive.CreateFile({'id': child['id']})
                if (binary):
                    dest_file = tempfile.NamedTemporaryFile(delete=False)
                    dest_file.close()
                    file.GetContentFile(dest_file.name)
                    b2 = open(dest_file.name, 'rb').read()
                    f = open(path, 'w+b')
                    f.write(b2)
                    f.close()
                else:
                    file.GetContentFile(path)
                print('Downloaded ', file_name)
                return True
        print('Not available to download ', file_name)
        return False